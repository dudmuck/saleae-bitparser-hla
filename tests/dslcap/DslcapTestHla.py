from saleae.analyzers import AnalyzerFrame


class TestHla:
    def decode(self, frame):
        if frame.type == 'result':
            mosi, miso = frame.data['mosi'][0], frame.data['miso'][0]
            return AnalyzerFrame('result', frame.start_time, frame.end_time,
                                 {'string': f'byte {mosi:02x}/{miso:02x}'})
