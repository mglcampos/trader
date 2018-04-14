
from datetime import datetime as dt
import time
import zmq

from htr.helpers.wrappers.kraken import Kraken
from htr.core.brokerage import BrokerHandler

class KrakenHandler(BrokerHandler):

    def __init__(self, context):
        """."""

        # Create REQ Socket
        self.reqSocket = context.socket(zmq.REQ)
        self.reqSocket.connect("tcp://localhost:5555")

        ## todo contemplate the case of lost internet connection

    def create_order(self, symbol, amount, price, side, ord_type):
        """Creates an Limit/Market/Stop order.

        Args:
        	symbol (str): Instrument.
        	amount (float): Order size.
        	price (float): Price.
        	side (str): Buy or sell.
        	type (str): Either “market” / “limit” / “stop” / “trailing-stop” / “fill-or-kill” / “exchange market” / “exchange limit” / “exchange stop” / “exchange trailing-stop” / “exchange fill-or-kill”. (type starting by “exchange ” are exchange orders, others are margin trading orders)
        """

        output = self.kraken.add_order(symbol, side, ord_type, price, amount)
        response = {'order_id' : -1, 'exchange' : 'Kraken', 'price' : price, 'timestamp' : time.mktime(dt.now().timetuple())}

        try:
            response['order_id'] = output['result']['txid'][0]

        except:
            response['order_id'] = -1

        return response

    def __get_tick(self, symbol):
        """."""

        return self.kraken.ticker(symbol)

    def ticker(self, symbol):
        """Retrieves last tick."""

        ## todo improve this
        tick = self.kraken.ticker(symbol)
        ## todo retrieve volume, save all into bd
        rsp = [tick['result'][s] for s in tick['result']][0]
        tick = {}
        tick['bid'] = rsp['b'][0]
        tick['ask'] = rsp['a'][0]
        tick['close'] = rsp['c'][0]
        tick['timestamp'] = time.mktime(dt.now().timetuple())

        return tick

    def get_server_time(self):
        """."""

        return self.kraken.server_time()

    def get_equity(self):
        """."""

        balances = self.kraken.trade_balance()
        return float(balances['result']['eb'])

    def get_available_units(self, symbol=None):
        """Returns the available units."""

        if symbol is None:
            return self.kraken.balance()['result']

        else:
            return float(self.kraken.balance()['result']['X'+symbol])

    def get_cash(self, symbol='USD'):
        """."""

        return float(self.kraken.trade_balance(symbol)['result']['mf'])

    def get_max_buy(self, pair):
        """."""

        tick = self.__get_tick(pair)['result']
        return 1 / float([tick[close]['c'][0] for close in tick][0]) * float(self.get_cash())

    def downsize_order(self, amount, pct=0.005):
        """."""

        slice = float(amount) * pct
        return amount - slice

    def get_max_sell(self, pair):
       """."""

       return self.get_available_units(pair[:3])

    # def get_symbols(self):
    #
    #     return self.KRAKEN.symbols()
    #
    # # def get_history(self, symbol, start, end):
    # #     raise NotImplementedError("Not implemented yet")
    #
    # def get_pending_orders(self):
    #
    #     return self.KRAKEN.active_orders()
    #
    # def status_order(self, order_id):
    #
    #     return self.KRAKEN.status_order(order_id)
    #
    # def get_open_positions(self):
    #
    #     ## todo test this
    #
    #     positions = []
    #
    #     balances = self.KRAKEN.balances()
    #     for wallet in balances:
    #         if float(wallet['available']) > 0.0 and wallet['type'] == 'exchange':
    #             positions.append(wallet)
    #
    #     return positions

# kraken = KrakenHandler('')
# # print(kraken.get_cash())
# # print(kraken.get_server_time())
# print(kraken.ticker('XRPUSD'))
