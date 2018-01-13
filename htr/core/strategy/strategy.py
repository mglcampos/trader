from abc import ABCMeta, abstractmethod

class Strategy(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def _load_data(self):
        raise NotImplementedError("Should implement load_data()!")

    @abstractmethod
    def _check_stop(self):
        raise NotImplementedError("Should implement evalute_data()!")

    @abstractmethod
    def calculate_signals(self):
        raise NotImplementedError("Should implement generate_signals()!")