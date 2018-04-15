
from datetime import datetime as dt
import time
import zmq

from htr.core.brokerage import BrokerHandler

class MetatraderHandler(BrokerHandler):

    def __init__(self, context):
        """."""

        context = zmq.Context()
        # Create REQ Socket
        self.reqSocket = context.socket(zmq.REQ)
        self.reqSocket.connect("tcp://localhost:5555")

    def create_order(self, symbol, amount, price, side, ord_type):
        """Creates an Limit/Market/Stop order.

        Args:
        	symbol (str): Instrument.
        	amount (float): Order size.
        	price (float): Price.
        	side (str): Buy or sell.
        	type (str): Either “market” / “limit” / “stop” / “trailing-stop” / “fill-or-kill” / “exchange market” / “exchange limit” / “exchange stop” / “exchange trailing-stop” / “exchange fill-or-kill”. (type starting by “exchange ” are exchange orders, others are margin trading orders)
        """

        # eurusd_buy_order = "TRADE|OPEN|0|EURUSD|0|50|50|Python-to-MT4"
        # eurusd_sell_order = "TRADE|OPEN|1|EURUSD|0|50|50|Python-to-MT4"
        # eurusd_closebuy_order = "TRADE|CLOSE|0|EURUSD|0|50|50|Python-to-MT4"
        # get_rates = "RATES|BTCUSD"

        if side == 'buy' :
            op = 0
        elif side == 'sell' :
            op = 1
        else:
            op = -1

        if ord_type == 'MKT-CLOSE':
            action = 'CLOSE'
        elif ord_type == 'MKT-OPEN':
            action = 'OPEN'
        else:
            action = ''

        order = 'TRADE|{}|{}|{}|{}|{}|{}|{}|{}'.format(action, op, symbol, 0, '', '', amount, '')

        # // 1) Trading
        #TRADE | ACTION | TYPE | SYMBOL | PRICE | SL | TP | Lots | TICKETID
        # // e.g.TRADE | OPEN | 1 | EURUSD | 0 | 50 | 50 | R - to - MetaTrader4 | 0.01 | 12345678

        # // 2.1) RATES | SYMBOL   -> Returns Current Bid / Ask
        # // 2.2) DATA | SYMBOL | TIMEFRAME | START_DATETIME | END_DATETIME

        # // OP_BUY = 0
        # // OP_SELL = 1
        # // OP_BUYLIMIT = 2
        # // OP_SELLLIMIT = 3
        # // OP_BUYSTOP = 4
        # // OP_SELLSTOP = 5

        return self.remote_send(self.reqSocket, order)

    def remote_send(self, socket, data):

        try:
            socket.send_string(data)
            msg = socket.recv_string()
            print(msg)
            return msg

        except zmq.Again as e:
            print("Waiting for PUSH from MetaTrader 4..")

    def __get_tick(self, symbol):
        """."""

        return "RATES|"+symbol

    # def ticker(self, symbol):
    #     """Retrieves last tick."""
    #
    #     ## todo improve this
    #     tick = self.kraken.ticker(symbol)
    #     ## todo retrieve volume, save all into bd
    #     rsp = [tick['result'][s] for s in tick['result']][0]
    #     tick = {}
    #     tick['bid'] = rsp['b'][0]
    #     tick['ask'] = rsp['a'][0]
    #     tick['close'] = rsp['c'][0]
    #     tick['timestamp'] = time.mktime(dt.now().timetuple())
    #
    #     return tick

    # def get_server_time(self):
    #     """."""
    #
    #     return self.kraken.server_time()

    # def get_equity(self):
    #     """."""
    #
    #     balances = self.kraken.trade_balance()
    #     return float(balances['result']['eb'])
    #
    # def get_available_units(self, symbol=None):
    #     """Returns the available units."""
    #
    #     if symbol is None:
    #         return self.kraken.balance()['result']
    #
    #     else:
    #         return float(self.kraken.balance()['result']['X'+symbol])
    #
    def get_cash(self, symbol='USD'):
        """."""
        ## todo not using symbol

        # ACCOUNT | 1  # balance
        # ACCOUNT | 2  # current P/L
        # ACCOUNT | 3  # equity
        # ACCOUNT | 4  # margin used
        # ACCOUNT | 5  # free margin
        # ACCOUNT | 6  # margin level
        # ACCOUNT | 7  # margin call level
        # ACCOUNT | 8  # margin stop out level
        # ACCOUNT | 0  # all of the above.

        msg = 'ACCOUNT|{}'.format('5')
        return self.remote_send(self.reqSocket, msg)
    #
    # def get_max_buy(self, pair):
    #     """."""
    #
    #     tick = self.__get_tick(pair)['result']
    #     return 1 / float([tick[close]['c'][0] for close in tick][0]) * float(self.get_cash())
    #
    # def downsize_order(self, amount, pct=0.005):
    #     """."""
    #
    #     slice = float(amount) * pct
    #     return amount - slice
    #
    # def get_max_sell(self, pair):
    #    """."""
    #
    #    return NotImplementedError

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
