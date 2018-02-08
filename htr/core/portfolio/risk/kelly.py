
from htr.core.portfolio.risk import RiskHandler

class Kelly(RiskHandler):

    def __init__(self, context):
        """

        Args:
        	curr_positions (dict): Contains the curr_positions of the portfolio
        """
        self.current_positions = None
        self.context = context

    def update_current_positions(self, current_positions):
        self.current_positions = current_positions

    def calculate_trade(self, positions_dict, signal, close_value):
        # todo sacar isto - symbol, strength, dir, pos, pl_ratio
        self.update_current_positions(positions_dict)
       ## todo assert there is enough units to generate quantity
        return self._calculate_quantity(close_value, signal)


    def evaluate_group_trade(self, positions_dict, signal):
        # todo sacar isto - symbol, strength, dir, pos, pl_ratio
        raise NotImplementedError('Should Implement')

    def _calculate_quantity(self, close_value, signal):
        """Calculate quantity for base currency in pair (e.g. for XRPUSD calculates XRP amount)"""

        quantity = 0
        if signal.signal_type == 'LONG':
            ## todo assert quantity is < cash available
            quantity = 50 * (1/close_value) # todo change this

        elif signal.signal_type == 'EXIT':
            print(self.current_positions)
            quantity = self.current_positions[signal.symbol]

       ##kelly formula = hit rate and takeprofit / stop loss
       # K = ( PxB – (1–P) ) / B
       # invest always below ( K/2 / 100 ) * holdings

       #self.current_positions
        print('\nQUANTITY CALCULATED: ', quantity, signal)
        return quantity




