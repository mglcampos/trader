
class DataObject():
    """
    Gives structure to data
    """

    def __init__(self, start_date, bars, **kwargs):
        """
        Args:
            start_date: (datetime) Start date for available data
            bars: (dict) Dict with symbols as keys containing the available data
        """
        self.start_date = start_date
        self.bars = bars
        for k in kwargs:
            setattr(self,k,kwargs[k])




