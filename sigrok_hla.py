#!/usr/bin/env python3
"""
Live SPI capture with HLA decoding via sigrok-cli or Saleae Logic 2 automation.

Two backends:
  sigrok:  Uses sigrok-cli SPI protocol decoder (for sigrok-supported hardware)
  saleae:  Uses Saleae Logic 2 automation API (requires Logic 2 app with automation enabled)

Examples:
  # Saleae Logic 2 automation - single SPI port (channels 0-3)
  ./sigrok_hla.py --hla-path ~/HLA/saleae_lr2021 --saleae \\
      --spi 0,1,2,3 --samplerate 4M --time 5

  # Saleae Logic 2 - dual SPI port
  ./sigrok_hla.py --hla-path ~/HLA/saleae_lr2021 --saleae \\
      --spi 0,1,2,3 --spi 4,5,6,7 --samplerate 4M --time 10

  # Saleae Logic 2 - manual capture (press Ctrl-C to stop)
  ./sigrok_hla.py --hla-path ~/HLA/saleae_lr2021 --saleae \\
      --spi 0,1,2,3 --samplerate 4M

  # sigrok backend - from binary file
  ./sigrok_hla.py --hla-path ~/HLA/saleae_lr2021 -d fx2lafw \\
      -C 0=SCLK,1=MISO,2=MOSI,3=nSS --spi SCLK,MISO,MOSI,nSS --samplerate 4M

  # sigrok backend - from input file
  ./sigrok_hla.py --hla-path ~/HLA/saleae_lr2021 \\
      -i capture.bin -I binary:numchannels=4:samplerate=1000000 \\
      --spi 0,1,2,3
"""

import argparse
import contextlib
import csv
import heapq
import importlib
import json
import re
import subprocess
import sys
import os
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from saleae.analyzers import AnalyzerFrame


def _load_hla_class(hla_path):
    """Load the HLA class from the given directory.

    Reads extension.json to discover the entry point (module.ClassName),
    falling back to 'HighLevelAnalyzer.Hla' for older HLA layouts.
    """
    ext_json = os.path.join(hla_path, 'extension.json')
    module_name = 'HighLevelAnalyzer'
    class_name = 'Hla'

    if os.path.isfile(ext_json):
        with open(ext_json) as f:
            ext = json.load(f)
        extensions = ext.get('extensions', {})
        for _label, info in extensions.items():
            if info.get('type') == 'HighLevelAnalyzer':
                entry = info['entryPoint']  # e.g. "lr20xx_hla_main.LR20xxProtocolAnalyzer"
                module_name, class_name = entry.rsplit('.', 1)
                break

    if hla_path not in sys.path:
        sys.path.insert(0, hla_path)

    try:
        mod = importlib.import_module(module_name)
    except ImportError as e:
        print(f"Failed to import HLA module '{module_name}' from {hla_path}: {e}",
              file=sys.stderr)
        sys.exit(1)

    cls = getattr(mod, class_name, None)
    if cls is None:
        print(f"HLA module '{module_name}' has no class '{class_name}'", file=sys.stderr)
        sys.exit(1)

    print(f"HLA: {module_name}.{class_name}", file=sys.stderr)
    return cls


# ---------------------------------------------------------------------------
# Common HLA helpers
# ---------------------------------------------------------------------------

def _format_hla_result(result):
    """Format an HLA result frame into a display string.

    Supports HLAs that return a 'string' field (e.g. saleae_lr2021) and
    HLAs that return 'command_name'/'parsed_params'/'chip_status' fields
    (e.g. lr20xx-saleae-protocol-analyzer).
    """
    data = result.data
    if 'string' in data and data['string']:
        return data['string']
    # Build from structured fields
    parts = []
    if data.get('command_name'):
        parts.append(data['command_name'])
    if data.get('parsed_params'):
        parts.append(data['parsed_params'])
    if data.get('chip_status'):
        parts.append(f"[{data['chip_status']}]")
    if parts:
        return ' '.join(parts)
    # Last resort: show all non-empty values
    return ' '.join(str(v) for v in data.values() if v)


def _feed_hla(hla, port_name, frame, results_heap, seq, show_port_prefix,
              flush=True):
    """Feed a frame to the HLA and queue any result for ordered output.

    An HLA timestamps its result at the *start* of the transaction but only
    returns it once the transaction ends, so results are produced slightly
    out of order. Callers that want chronological output pass flush=False
    and drain the heap themselves once a batch is complete.
    """
    try:
        with contextlib.redirect_stdout(sys.stderr):
            result = hla.decode(frame)
    except Exception as e:
        error_msg = f"*** DECODE ERROR: {type(e).__name__}: {e} ***"
        port_prefix = f"[{port_name}] " if show_port_prefix else ""
        heapq.heappush(results_heap, (frame.start_time, seq, port_name,
            f"{frame.start_time:.9f}: {port_prefix}{error_msg}"))
        return

    if result is not None:
        msg = _format_hla_result(result)
        port_prefix = f"[{port_name}] " if show_port_prefix else ""
        heapq.heappush(results_heap, (result.start_time, seq, port_name,
            f"{result.start_time:.9f}: {port_prefix}{msg}"))

    if flush:
        _flush_results(results_heap)


def _flush_results_until(results_heap, watermark):
    """Print queued results older than ``watermark``, in timestamp order.

    Holding back the newest results lets a transaction that straddles a
    chunk boundary settle before anything after it is printed, so the
    output stays chronological across the whole stream.
    """
    while results_heap and results_heap[0][0] < watermark:
        _, _, _, text = heapq.heappop(results_heap)
        try:
            print(text, flush=True)
        except BrokenPipeError:
            raise


def _flush_results(results_heap):
    """Print all queued results in timestamp order."""
    while results_heap:
        _, _, _, text = heapq.heappop(results_heap)
        try:
            print(text, flush=True)
        except BrokenPipeError:
            # Downstream consumer closed (e.g. head, grep -m), exit quietly
            devnull = os.open(os.devnull, os.O_WRONLY)
            os.dup2(devnull, sys.stdout.fileno())
            raise SystemExit(0)


# ---------------------------------------------------------------------------
# Saleae Logic 2 automation backend
# ---------------------------------------------------------------------------

CPOL_SETTINGS = {
    0: 'Clock is Low when inactive (CPOL = 0)',
    1: 'Clock is High when inactive (CPOL = 1)',
}
CPHA_SETTINGS = {
    0: 'Data is Valid on Clock Leading Edge (CPHA = 0)',
    1: 'Data is Valid on Clock Trailing Edge (CPHA = 1)',
}


def run_saleae_backend(args, spi_ports, port_name_list):
    """Run capture and HLA decode using Saleae Logic 2 automation API."""
    # Import saleae automation (avoid our local mock saleae package)
    orig_path = sys.path[:]
    sys.path = [p for p in sys.path if 'saleae-binparser' not in p]
    # Remove cached mock saleae module so the real one can be found
    for key in list(sys.modules.keys()):
        if key == 'saleae' or key.startswith('saleae.'):
            del sys.modules[key]
    try:
        from saleae.automation import Manager
        from saleae.automation.manager import (
            LogicDeviceConfiguration, TimedCaptureMode, ManualCaptureMode,
            CaptureConfiguration,
        )
        from saleae.automation.capture import DataTableExportConfiguration, RadixType
    finally:
        sys.path = orig_path

    m = Manager.connect(port=args.saleae_port)
    devs = m.get_devices()
    if not devs:
        print("No Saleae devices found. Is the device connected and Logic 2 running?",
              file=sys.stderr)
        sys.exit(1)

    dev = devs[0]
    print(f"Device: {dev.device_type.name} (id={dev.device_id})", file=sys.stderr)

    # Collect all digital channels needed
    all_channels = set()
    for port in spi_ports:
        for ch in [port['clk'], port['miso'], port['mosi'], port['cs']]:
            all_channels.add(ch)

    samplerate = int(parse_samplerate(args.samplerate)) if args.samplerate else 4_000_000

    dev_config = LogicDeviceConfiguration(
        enabled_digital_channels=sorted(all_channels),
        digital_sample_rate=samplerate,
    )

    if args.time:
        duration = parse_duration(args.time)
        cap_mode = TimedCaptureMode(duration_seconds=duration)
        print(f"Timed capture: {duration}s at {samplerate/1e6:.1f} MSa/s", file=sys.stderr)
    else:
        cap_mode = ManualCaptureMode()
        print(f"Manual capture at {samplerate/1e6:.1f} MSa/s (Ctrl-C to stop)", file=sys.stderr)

    cap_kwargs = dict(capture_mode=cap_mode)
    if args.buffer_size:
        cap_kwargs['buffer_size_megabytes'] = args.buffer_size
    cap_config = CaptureConfiguration(**cap_kwargs)

    import time as _time

    print("Starting capture...", file=sys.stderr)
    cap = m.start_capture(
        device_configuration=dev_config,
        device_id=dev.device_id,
        capture_configuration=cap_config,
    )

    export_path = None
    try:
        if isinstance(cap_mode, ManualCaptureMode):
            if args.stop_file:
                # Remove stale stop file
                if os.path.exists(args.stop_file):
                    os.unlink(args.stop_file)
                print(f"Capturing... touch {args.stop_file} to stop", file=sys.stderr)
            else:
                print("Capturing... press Ctrl-C to stop", file=sys.stderr)
            try:
                while True:
                    if args.stop_file and os.path.exists(args.stop_file):
                        print("Stop file detected.", file=sys.stderr)
                        os.unlink(args.stop_file)
                        break
                    _time.sleep(0.25)
            except KeyboardInterrupt:
                print("\nStopping capture...", file=sys.stderr)
            cap.stop()
        else:
            cap.wait()

        print("Capture complete.", file=sys.stderr)

        # Add SPI analyzers for each port
        spi_analyzers = []
        for port, pname in zip(spi_ports, port_name_list):
            spi = cap.add_analyzer('SPI', label=pname, settings={
                'Clock': port['clk'],
                'MISO': port['miso'],
                'MOSI': port['mosi'],
                'Enable': port['cs'],
                'Bits per Transfer': '8 Bits per Transfer (Standard)',
                'Significant Bit': 'Most Significant Bit First (Standard)',
                'Clock State': CPOL_SETTINGS[args.cpol],
                'Clock Phase': CPHA_SETTINGS[args.cpha],
                'Enable Line': 'Enable line is Active Low (Standard)',
            })
            spi_analyzers.append((pname, spi))
            print(f"  Added SPI analyzer: {pname} (clk={port['clk']} miso={port['miso']} "
                  f"mosi={port['mosi']} cs={port['cs']})", file=sys.stderr)

        # Export data table to CSV
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False, mode='w') as tmp:
            export_path = tmp.name

        analyzer_handles = [spi for _, spi in spi_analyzers]
        export_configs = [DataTableExportConfiguration(h, RadixType.HEXADECIMAL) for h in analyzer_handles]
        print("Exporting SPI data...", file=sys.stderr)
        t0 = _time.monotonic()
        cap.export_data_table(export_path, export_configs)
        t1 = _time.monotonic()

        # Check if there's any data
        n_lines = sum(1 for _ in open(export_path)) - 1  # subtract header
        print(f"Exported {n_lines} rows in {t1-t0:.1f}s", file=sys.stderr)

        if n_lines <= 0:
            print("No SPI transactions captured.", file=sys.stderr)
        else:
            print("Processing...", file=sys.stderr)
            _process_saleae_csv(export_path, args, spi_analyzers, port_name_list)
            print(f"Done in {_time.monotonic()-t1:.1f}s", file=sys.stderr)

    except KeyboardInterrupt:
        print("\nInterrupted, cleaning up...", file=sys.stderr)
        try:
            cap.stop()
        except Exception:
            pass
    finally:
        if export_path and os.path.exists(export_path):
            os.unlink(export_path)
        cap.close()
        m.close()
        print("Capture closed.", file=sys.stderr)


def _process_saleae_csv(csv_path, args, spi_analyzers, port_name_list):
    """Parse Saleae SPI data table CSV and feed frames to HLA."""
    # Restore our mock saleae.analyzers for the HLA import
    for key in list(sys.modules.keys()):
        if key == 'saleae' or key.startswith('saleae.'):
            del sys.modules[key]
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)

    # Import HLA
    Hla = _load_hla_class(args.hla_path)

    show_port_prefix = len(port_name_list) > 1

    # Create per-port HLA instances
    hla_map = {}  # port_name -> Hla instance
    for pname in port_name_list:
        hla_map[pname] = Hla()

    results_heap = []
    seq = 0
    hex_bufs = {pname: {'mosi': bytearray(), 'miso': bytearray()} for pname in port_name_list}

    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            pname = row['name'].strip()
            ftype = row['type'].strip()
            start_time = float(row['start_time'])

            if pname not in hla_map:
                # Multi-port: match by analyzer label
                # The CSV 'name' column contains the analyzer label we set
                if pname not in hla_map:
                    continue

            hla = hla_map[pname]

            if ftype == 'enable':
                frame = AnalyzerFrame('enable', start_time, start_time)
                hex_bufs[pname] = {'mosi': bytearray(), 'miso': bytearray()}
            elif ftype == 'disable':
                if args.hex:
                    buf = hex_bufs[pname]
                    if buf['mosi'] or buf['miso']:
                        prefix = f"  [{pname}] " if show_port_prefix else "  "
                        heapq.heappush(results_heap, (start_time, seq, pname,
                            f"{prefix}MOSI: {buf['mosi'].hex(' ')}\n"
                            f"{prefix}MISO: {buf['miso'].hex(' ')}"))
                        seq += 1
                frame = AnalyzerFrame('disable', start_time, start_time)
            elif ftype == 'result':
                miso_hex = row.get('miso', '').strip()
                mosi_hex = row.get('mosi', '').strip()
                miso_val = int(miso_hex, 16) if miso_hex else 0
                mosi_val = int(mosi_hex, 16) if mosi_hex else 0
                frame = AnalyzerFrame('result', start_time, start_time, {
                    'mosi': bytes([mosi_val]),
                    'miso': bytes([miso_val]),
                })
                hex_bufs[pname]['mosi'].append(mosi_val)
                hex_bufs[pname]['miso'].append(miso_val)
            else:
                continue

            _feed_hla(hla, pname, frame, results_heap, seq, show_port_prefix)
            seq += 1

    _flush_results(results_heap)


# ---------------------------------------------------------------------------
# sigrok backend
# ---------------------------------------------------------------------------

def resolve_channel_indices(args, spi_ports):
    """Map the SPI ports' channel names to logic bit indices.

    The numpy engine reads packed samples where bit N is logic channel N, so
    it needs indices rather than the names the SPI PD takes. Names come from
    -C (e.g. "0=SCLK,1=MISO,..."); bare numbers are accepted directly.
    """
    name_to_idx = {}
    if args.channels:
        for item in args.channels.split(','):
            item = item.strip()
            if '=' in item:
                idx, name = item.split('=', 1)
                idx = idx.strip()
                # Accept both "0=SCLK" and driver-native names like "D0=SCLK".
                digits = ''.join(c for c in idx if c.isdigit())
                if digits:
                    name_to_idx[name.strip()] = int(digits)

    def to_index(val, what):
        if val in name_to_idx:
            idx = name_to_idx[val]
        elif val.isdigit():
            idx = int(val)
        else:
            # Channel names are matched case-insensitively, like spi_hla.py.
            for name, i in name_to_idx.items():
                if name.lower() == val.lower():
                    idx = i
                    break
            else:
                raise SystemExit(
                    f"{what} '{val}': cannot map to a channel index. Name it in "
                    f"-C (e.g. -C 0={val},...) or give a channel number. "
                    f"Known: {', '.join(sorted(name_to_idx)) or '(none)'}")
        if idx > 7:
            raise SystemExit(
                f"{what} '{val}': channel index {idx} exceeds 7; the packed "
                "sample stream holds 8 channels.")
        return idx

    resolved = []
    for port in spi_ports:
        resolved.append({role: to_index(port[role], '--engine numpy')
                         for role in ('clk', 'miso', 'mosi', 'cs')})

    pins = []
    for val, what in ([(args.int_pin, '--int-pin')] if args.int_pin else []) + \
                     [(p, '--extra-pin') for p in args.extra_pin]:
        pins.append((val, to_index(val, what)))

    return resolved, pins


def build_sigrok_cmd(args, spi_ports):
    """Build the sigrok-cli command line from parsed arguments."""
    cmd = ['sigrok-cli']

    if args.driver:
        cmd += ['-d', args.driver]
    if args.input_file:
        cmd += ['-i', args.input_file]
        if args.input_format:
            cmd += ['-I', args.input_format]
    if args.channels:
        cmd += ['-C', args.channels]
    if args.transform:
        cmd += ['-T', args.transform]
    if args.samplerate:
        cmd += ['--config', f'samplerate={args.samplerate}']
    if args.samples:
        cmd += ['--samples', args.samples]
    if args.continuous:
        cmd += ['--continuous']
    if args.time:
        cmd += ['--time', args.time]

    if getattr(args, 'engine', 'srd') == 'numpy':
        # The numpy engine decodes the raw sample stream itself: no protocol
        # decoder is instantiated, so libsigrokdecode does no work at all.
        cmd += ['-O', 'binary']
        return cmd

    for port in spi_ports:
        spi_opts = [
            f"clk={port['clk']}",
            f"miso={port['miso']}",
            f"mosi={port['mosi']}",
            f"cs={port['cs']}",
            f"cpol={args.cpol}",
            f"cpha={args.cpha}",
        ]
        cmd += ['-P', 'spi:' + ':'.join(spi_opts)]

    cmd += ['--protocol-decoder-samplenum']
    # Use the SPI decoder's per-transfer annotations (one annotation per
    # real CS# assertion, containing the whole byte sequence) rather than
    # per-byte data annotations. This frames transactions by the actual
    # nSS line, matching Saleae Logic 2. Reconstructing transactions from
    # inter-byte timing gaps (the old approach) wrongly split a single
    # CS#-asserted transfer whenever the firmware paused mid-transfer
    # (e.g. DMA setup between a WriteRadioTxFifo opcode and its payload).
    cmd += ['-A', 'spi=miso-transfer:mosi-transfer']

    return cmd


# A transfer annotation line: "START-END spi-N: HH HH HH ..." (space-separated
# hex bytes covering one full CS#-asserted transfer).
LINE_RE = re.compile(r'^(\d+)-(\d+)\s+spi-(\d+):\s+([0-9A-Fa-f]+(?:\s+[0-9A-Fa-f]+)*)$')


def run_sigrok_numpy(args, spi_ports, port_name_list, samplerate):
    """Capture via sigrok-cli -O binary and decode with the vectorized engine.

    sigrok-cli is run without any protocol decoder; this reads its raw sample
    stream and decodes SPI in NumPy, which keeps up with live capture. A
    reader thread with a deep queue decouples the pipe from decode work so a
    GC pause never back-pressures sigrok-cli into a USB overrun.
    """
    import queue
    import threading

    import numpy as np

    from fast_spi import MultiPortDecoder, PinLogger

    chan, log_pins = resolve_channel_indices(args, spi_ports)
    ports = [dict(name=name, **idx) for name, idx in zip(port_name_list, chan)]
    for p in ports:
        print(f"  {p['name']}: bits clk={p['clk']} miso={p['miso']} "
              f"mosi={p['mosi']} cs={p['cs']}", file=sys.stderr)
    for name, bit in log_pins:
        print(f"  pin {name}: bit {bit}", file=sys.stderr)
    pin_logger = PinLogger(log_pins, samplerate) if log_pins else None

    Hla = _load_hla_class(args.hla_path)
    hla_map = {name: Hla() for name in port_name_list}
    show_port_prefix = len(port_name_list) > 1
    hex_bufs = {name: {'mosi': bytearray(), 'miso': bytearray()}
                for name in port_name_list}

    decoder = MultiPortDecoder(ports, samplerate, cpol=args.cpol, cpha=args.cpha)

    cmd = build_sigrok_cmd(args, spi_ports)
    print(f"Running: {' '.join(cmd)}", file=sys.stderr)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            bufsize=0)

    CHUNK = 1 << 22        # 4 MiB reads
    # Output is held back this long (in capture seconds) so a transaction
    # spanning a chunk boundary can finish before later events are printed.
    # Far longer than any SPI transfer, far shorter than a chunk.
    HOLD = 0.05
    q = queue.Queue(maxsize=64)

    def reader():
        try:
            while True:
                buf = proc.stdout.read(CHUNK)
                if not buf:
                    break
                q.put(buf)
        except Exception as e:                          # pragma: no cover
            print(f"[reader] {e}", file=sys.stderr)
        finally:
            q.put(None)

    t = threading.Thread(target=reader, daemon=True)
    t.start()

    results_heap = []
    seq = 0
    first = True

    want_hex = args.hex

    def handle(name, frame):
        nonlocal seq
        if not want_hex:
            pass
        elif frame.type == 'enable':
            hex_bufs[name] = {'mosi': bytearray(), 'miso': bytearray()}
        elif frame.type == 'result':
            hex_bufs[name]['mosi'] += frame.data['mosi']
            hex_bufs[name]['miso'] += frame.data['miso']
        if frame.type == 'disable' and want_hex:
            buf = hex_bufs[name]
            if buf['mosi'] or buf['miso']:
                prefix = f"  [{name}] " if show_port_prefix else "  "
                heapq.heappush(results_heap, (frame.start_time, seq, name,
                    f"{prefix}MOSI: {buf['mosi'].hex(' ')}\n"
                    f"{prefix}MISO: {buf['miso'].hex(' ')}"))
                seq += 1
        _feed_hla(hla_map[name], name, frame, results_heap, seq,
                  show_port_prefix, flush=False)
        seq += 1

    def process(chunk_events):
        """Emit one chunk's events in timestamp order.

        Each port is decoded over the whole chunk before the next, so the
        per-port streams have to be merged here to read chronologically --
        which is also what makes a logged pin edge appear at the point in
        the traffic where it actually happened.
        """
        nonlocal seq
        chunk_events.sort(key=lambda e: e[0])
        for _, kind, payload in chunk_events:
            if kind == 'frame':
                handle(*payload)
            else:
                t, name, edge = payload
                heapq.heappush(results_heap, (t, seq, name,
                    f"{t:.9f}: [{name}] {edge}"))
                seq += 1
        # Results are queued, not printed, while the chunk is processed.
        # Everything older than the newest event minus HOLD is now settled:
        # a transaction still open across the chunk boundary can only carry
        # a timestamp newer than that.
        if chunk_events:
            _flush_results_until(results_heap, chunk_events[-1][0] - HOLD)

    try:
        while True:
            buf = q.get()
            if buf is None:
                break
            if first:
                # "-O binary" to a pipe prepends a "META samplerate: N\n" line.
                if buf.startswith(b'META'):
                    nl = buf.find(b'\n')
                    if nl >= 0:
                        buf = buf[nl + 1:]
                first = False
            chunk = np.frombuffer(buf, dtype=np.uint8)
            events = [(f.start_time, 'frame', (n, f))
                      for n, f in decoder.feed(chunk)]
            if pin_logger is not None:
                events += [(t, 'pin', (t, name, edge))
                           for t, name, edge in pin_logger.feed(chunk)]
            process(events)
        process([(f.start_time, 'frame', (n, f)) for n, f in decoder.end()])
    except KeyboardInterrupt:
        print("\nInterrupted", file=sys.stderr)
    finally:
        _flush_results(results_heap)
        proc.terminate()
        proc.wait()
        stderr_out = proc.stderr.read().decode(errors='replace')
        if stderr_out.strip():
            print(f"[sigrok stderr] {stderr_out.strip()}", file=sys.stderr)


def run_sigrok_backend(args, spi_ports, port_name_list):
    """Run capture and HLA decode using sigrok-cli."""
    samplerate = parse_samplerate(args.samplerate) if args.samplerate else 1e6
    if args.input_format and 'samplerate=' in args.input_format:
        for part in args.input_format.split(':'):
            if part.startswith('samplerate='):
                samplerate = parse_samplerate(part.split('=', 1)[1])
                break
    print(f"Sample rate: {samplerate:.0f} Hz", file=sys.stderr)

    if getattr(args, 'engine', 'srd') == 'numpy':
        run_sigrok_numpy(args, spi_ports, port_name_list, samplerate)
        return

    if args.int_pin or args.extra_pin:
        raise SystemExit('--int-pin/--extra-pin need the numpy engine '
                         '(libsigrokdecode only reports decoder output); '
                         'drop --engine srd.')

    # Import HLA
    Hla = _load_hla_class(args.hla_path)

    hla_instances = {}
    port_names = {}
    for i, name in enumerate(port_name_list):
        spi_id = i + 1
        hla_instances[spi_id] = Hla()
        port_names[spi_id] = name

    cmd = build_sigrok_cmd(args, spi_ports)
    print(f"Running: {' '.join(cmd)}", file=sys.stderr)

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, bufsize=1)
    try:
        parse_sigrok_output(proc, samplerate, hla_instances, port_names, args.hex)
    except KeyboardInterrupt:
        print("\nInterrupted", file=sys.stderr)
    finally:
        proc.terminate()
        proc.wait()
        stderr_out = proc.stderr.read()
        if stderr_out.strip():
            print(f"[sigrok stderr] {stderr_out.strip()}", file=sys.stderr)


def parse_sigrok_output(proc, samplerate, hla_instances, port_names, hex_mode):
    """Parse sigrok-cli SPI per-transfer annotations and feed to HLA instances.

    The SPI decoder is run with ``-A spi=miso-transfer:mosi-transfer``, so each
    CS#-asserted transfer yields exactly two annotation lines at the same
    sample range: the MISO transfer first (annotation class 5), then the MOSI
    transfer (class 6) -- the SPI decoder emits them in that order. We pair
    them by (spi_id, start, end) and feed each complete transfer to the HLA as
    one enable/result/disable sequence. Transactions are thus framed by the
    real nSS line, exactly like Saleae Logic 2 -- no inter-byte timing-gap
    heuristic (which used to split a transfer whenever the firmware paused
    mid-transfer, e.g. between a WriteRadioTxFifo opcode and its payload).
    """
    show_port_prefix = len(port_names) > 1

    # Per port: the first (MISO) half of a transfer awaiting its MOSI half,
    # keyed by (start_sample, end_sample).
    pending = {spi_id: {} for spi_id in hla_instances}

    results_heap = []
    seq = 0

    def hexbytes(s):
        return bytes(int(x, 16) for x in s.split())

    def emit_transfer(spi_id, start_sample, end_sample, miso_bytes, mosi_bytes):
        """Feed one complete CS#-framed transfer to its HLA instance."""
        nonlocal seq
        hla = hla_instances[spi_id]
        pname = port_names[spi_id]
        start_time = start_sample / samplerate
        end_time = end_sample / samplerate

        # nSS falling edge: start of transfer.
        _feed_hla(hla, pname, AnalyzerFrame('enable', start_time, start_time),
                  results_heap, seq, show_port_prefix)
        seq += 1
        # Whole transfer in a single result frame (the HLA accumulates bytes).
        _feed_hla(hla, pname, AnalyzerFrame('result', start_time, end_time,
                  {'mosi': mosi_bytes, 'miso': miso_bytes}),
                  results_heap, seq, show_port_prefix)
        seq += 1
        if hex_mode and (mosi_bytes or miso_bytes):
            prefix = f"  [{pname}] " if show_port_prefix else "  "
            heapq.heappush(results_heap, (end_time, seq, pname,
                f"{prefix}MOSI: {mosi_bytes.hex(' ')}\n"
                f"{prefix}MISO: {miso_bytes.hex(' ')}"))
            seq += 1
        # nSS rising edge: end of transfer -> the HLA decodes here.
        _feed_hla(hla, pname, AnalyzerFrame('disable', end_time, end_time),
                  results_heap, seq, show_port_prefix)
        seq += 1

    for raw_line in proc.stdout:
        line = raw_line.strip()
        m = LINE_RE.match(line)
        if not m:
            if line:
                print(f"[sigrok] {line}", file=sys.stderr)
            continue

        start_sample = int(m.group(1))
        end_sample = int(m.group(2))
        spi_id = int(m.group(3))
        hex_str = m.group(4).strip()

        if spi_id not in pending:
            print(f"Warning: unexpected spi-{spi_id} in output", file=sys.stderr)
            continue

        key = (start_sample, end_sample)
        port_pending = pending[spi_id]
        if key not in port_pending:
            # First line for this transfer == MISO transfer (emitted first).
            port_pending[key] = hexbytes(hex_str)
        else:
            # Second line == MOSI transfer; the pair is complete.
            miso_bytes = port_pending.pop(key)
            emit_transfer(spi_id, start_sample, end_sample,
                          miso_bytes, hexbytes(hex_str))

    # Flush transfers that produced only one annotation (e.g. a single-channel
    # SPI, or a transfer truncated at end-of-capture).
    for spi_id, port_pending in pending.items():
        for (start_sample, end_sample), miso_bytes in sorted(port_pending.items()):
            emit_transfer(spi_id, start_sample, end_sample, miso_bytes, b'')

    _flush_results(results_heap)


# ---------------------------------------------------------------------------
# Common utilities
# ---------------------------------------------------------------------------

def parse_samplerate(s):
    """Parse a sample rate string like '4M', '1000000', '500k' to Hz."""
    s = s.strip().upper()
    multipliers = {'K': 1e3, 'M': 1e6, 'G': 1e9}
    for suffix, mult in multipliers.items():
        if s.endswith(suffix):
            return float(s[:-1]) * mult
    return float(s)


def parse_duration(s):
    """Parse a duration string like '5s', '100ms', '5' to seconds."""
    s = s.strip().lower()
    if s.endswith('ms'):
        return float(s[:-2]) / 1000
    if s.endswith('s'):
        return float(s[:-1])
    return float(s)


def parse_spi_port(spec):
    """Parse an --spi spec into a dict.

    For sigrok: CLK_NAME,MISO_NAME,MOSI_NAME,CS_NAME (channel names)
    For saleae: CLK_CH,MISO_CH,MOSI_CH,CS_CH (channel numbers)
    """
    parts = spec.split(',')
    if len(parts) != 4:
        raise argparse.ArgumentTypeError(
            f"Expected CLK,MISO,MOSI,CS but got: {spec}")
    # Try to parse as integers (saleae mode) or keep as strings (sigrok mode)
    parsed = []
    for p in parts:
        try:
            parsed.append(int(p))
        except ValueError:
            parsed.append(p)
    return {'clk': parsed[0], 'miso': parsed[1], 'mosi': parsed[2], 'cs': parsed[3]}


def main():
    parser = argparse.ArgumentParser(
        description='Live SPI capture with HLA decoding (sigrok or Saleae Logic 2)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)

    # Backend selection
    parser.add_argument('--saleae', action='store_true',
                        help='Use Saleae Logic 2 automation API (requires Logic 2 with automation enabled)')
    parser.add_argument('--saleae-port', type=int, default=10430,
                        help='Saleae automation server port (default: 10430)')

    # sigrok-specific options
    parser.add_argument('-d', '--driver', type=str, default=None,
                        help='[sigrok] Driver (e.g., fx2lafw, saleae-logic16)')
    parser.add_argument('-i', '--input-file', type=str, default=None,
                        help='[sigrok] Input file instead of live capture')
    parser.add_argument('-I', '--input-format', type=str, default=None,
                        help='[sigrok] Input format (e.g., binary:numchannels=4:samplerate=1000000)')
    parser.add_argument('-C', '--channels', type=str, default=None,
                        help='[sigrok] Channel list (e.g., 0=SCLK,1=MISO,2=MOSI,3=nSS)')
    parser.add_argument('--samples', type=str, default=None,
                        help='[sigrok] Number of samples to capture')
    parser.add_argument('--continuous', action='store_true',
                        help='[sigrok] Continuous (streaming) capture')

    # Common options
    parser.add_argument('--spi', action='append', metavar='CLK,MISO,MOSI,CS',
                        help='SPI port (repeatable). Channel numbers for --saleae, '
                             'channel names for sigrok. '
                             'E.g., --spi 0,1,2,3 --spi 4,5,6,7')
    parser.add_argument('--samplerate', type=str, default=None,
                        help='Sample rate (e.g., 4M, 1000000)')
    parser.add_argument('--time', type=str, default=None,
                        help='Capture duration (e.g., 5s, 100ms)')
    parser.add_argument('-T', '--transform', type=str, default=None,
                        metavar='MODULE[:OPT=VAL...]',
                        help='[sigrok] Transform module applied before decoding, '
                             'e.g. deglitch:channels=SCLK,SCLK_B:clock_period=2.5:frame_pulses=8')
    parser.add_argument('--int-pin', type=str, default=None, metavar='NAME',
                        help='[sigrok] Log transitions of an interrupt pin, '
                             'interleaved with decoded traffic (e.g. --int-pin int)')
    parser.add_argument('--extra-pin', type=str, default=[], action='append',
                        metavar='NAME',
                        help='[sigrok] Log transitions of an extra pin '
                             '(repeatable, e.g. --extra-pin busy --extra-pin dbg)')
    parser.add_argument('--engine', choices=['numpy', 'srd'], default='numpy',
                        help='[sigrok] SPI decode engine: "numpy" (default, '
                             'vectorized, keeps up with live capture) or "srd" '
                             '(libsigrokdecode SPI PD, ~20x slower, reference)')
    parser.add_argument('--cpol', type=int, default=0, choices=[0, 1])
    parser.add_argument('--cpha', type=int, default=0, choices=[0, 1])
    parser.add_argument('--hla-path', type=str, required=True,
                        help='Path to HLA directory containing HighLevelAnalyzer.py')
    parser.add_argument('--hex', action='store_true',
                        help='Print MOSI/MISO hex bytes before each decoded transaction')
    parser.add_argument('--buffer-size', type=int, default=None, metavar='MB',
                        help='[saleae] Capture buffer size in MB (limits memory usage)')
    parser.add_argument('--stop-file', type=str, default=None, metavar='PATH',
                        help='[saleae] Stop capture when this file appears (for scripted tests)')

    args = parser.parse_args()

    if not args.spi:
        parser.error('At least one --spi CLK,MISO,MOSI,CS is required')

    spi_ports = [parse_spi_port(spec) for spec in args.spi]

    if len(spi_ports) == 1:
        port_name_list = ['SPI']
    else:
        suffixes = ['', '_B', '_C', '_D', '_E', '_F', '_G', '_H']
        port_name_list = [f'SPI{s}' for s in suffixes[:len(spi_ports)]]

    print(f"SPI port(s): {len(spi_ports)}", file=sys.stderr)
    for port, name in zip(spi_ports, port_name_list):
        print(f"  {name}: clk={port['clk']} miso={port['miso']} "
              f"mosi={port['mosi']} cs={port['cs']}", file=sys.stderr)

    if args.saleae:
        run_saleae_backend(args, spi_ports, port_name_list)
    else:
        if not args.driver and not args.input_file:
            parser.error('sigrok backend requires -d (driver) or -i (input file). '
                         'Use --saleae for Saleae Logic 2.')
        run_sigrok_backend(args, spi_ports, port_name_list)


if __name__ == '__main__':
    main()
