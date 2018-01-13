
from abc import ABC, abstractmethod

class RiskHandler(ABC):

    @abstractmethod
    def calculate_trade(self, positions, event):
        """Evaluates if the signal should be converted to order."""

        raise NotImplementedError('Should implement')

    @abstractmethod
    def evaluate_group_trade(self, positions, event):
        """Evaluates if the group signal should be converted to order."""

        raise NotImplementedError('Should implement')
