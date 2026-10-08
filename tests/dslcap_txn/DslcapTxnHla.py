from saleae.analyzers import AnalyzerFrame


class TxnHla:
    """Reports once per transaction on nSS release, timestamped at its start,
    like the LR2021 HLA."""
    def __init__(self):
        self.start = None
        self.mosi = b''

    def decode(self, frame):
        if frame.type == 'enable':
            self.start, self.mosi = frame.start_time, b''
        elif frame.type == 'result':
            self.mosi += frame.data['mosi']
        elif frame.type == 'disable':
            return AnalyzerFrame('result', self.start, frame.end_time,
                                 {'string': f'txn {self.mosi.hex()}'})
