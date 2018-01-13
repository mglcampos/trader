
from htr.core.portfolio.risk import RiskHandler

class Kelly(RiskHandler):

    def __init__(self):
        """

        Args:
        	curr_positions (dict): Contains the curr_positions of the portfolio
        """
        self.current_positions = None

    def update_current_positions(self, current_positions):
        self.current_positions = current_positions

    def calculate_trade(self, positions_dict, signal, close_value):
        # todo sacar isto - symbol, strength, dir, pos, pl_ratio
        self.update_current_positions(positions_dict)
        #
        #
        #
        #
        # todo if something
        return self._calculate_quantity(close_value)


    def evaluate_group_trade(self, positions_dict, signal):
        # todo sacar isto - symbol, strength, dir, pos, pl_ratio
        raise NotImplementedError('Should Implement')

    def _calculate_quantity(self, close_value):

       quantity = 100 * (1/close_value) # todo change this

       ##kelly formula = hit rate and takeprofit / stop loss
       # K = ( PxB – (1–P) ) / B
       # invest always below ( K/2 / 100 ) * holdings

       #self.current_positions

       return quantity




