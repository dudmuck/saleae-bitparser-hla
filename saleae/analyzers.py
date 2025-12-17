# Mock saleae.analyzers module for standalone HLA execution

class AnalyzerFrame:
    """Mock AnalyzerFrame that mimics Saleae's SPI analyzer output."""
    def __init__(self, frame_type, start_time, end_time, data=None):
        self.type = frame_type
        self.start_time = start_time
        self.end_time = end_time
        self.data = data if data is not None else {}

class HighLevelAnalyzer:
    """Base class for High Level Analyzers."""
    pass

# Settings classes (not used in standalone mode, but needed for import)
class StringSetting:
    def __init__(self, *args, **kwargs):
        pass

class NumberSetting:
    def __init__(self, *args, **kwargs):
        pass

class ChoicesSetting:
    def __init__(self, *args, **kwargs):
        pass
