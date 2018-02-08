
from htr.helpers.wrappers.bitfinex import Bitfinex
from htr.core.brokerage import BrokerHandler

class BitfinexHandler(BrokerHandler):

    def __init__(self, context):
        """."""

        ## todo change this to retrieve keys from config file
        self.BITFINEX_KEY = "yNf6lIdg5EzRDnzAqO7Oncl2udMuvCMtpyhM47SQya3"
        self.BITFINEX_SECRET = 'B5KLhpoLfAs3vlUgKZt1x5NidHQhf1xyYL3tPenNI9c'

        ## todo change this to create a list of wrapper instances
        self.bitfinex = Bitfinex.Auth(key=self.BITFINEX_KEY, secret=self.BITFINEX_SECRET)

    def create_order(self, symbol, amount, price, side, ord_type):
        """Creates an Limit/Market/Stop order.

        Args:
        	symbol (str): Instrument.
        	amount (float): Order size.
        	price (float): Price.
        	side (str): Buy or sell.
        	type (str): Either “market” / “limit” / “stop” / “trailing-stop” / “fill-or-kill” / “exchange market” / “exchange limit” / “exchange stop” / “exchange trailing-stop” / “exchange fill-or-kill”. (type starting by “exchange ” are exchange orders, others are margin trading orders)
        """

        ## todo test this
        output = self.bitfinex.place_order(amount,
                    price,
                    side,
                    ord_type,
                    symbol)

        # print(reports)
        return output


    def get_tick(self, symbol):
        """Retrieves last tick."""

        return self.bitfinex.ticker(symbol)

    ## todo find the usd equivalent
    def get_equity(self, currency='usd', n=1):
        pass
    #     """Returns all the equity."""
    #
    #     return self.bitfinex.balance_history(currency)[-1]

    def get_available_units(self, symbol=None):
        """Returns the available units."""

        if symbol is None:
            return self.bitfinex.balances()

        else:
            balances = self.bitfinex.balances()
            for wallet in balances:
                if wallet['currency'] == symbol and wallet['type'] == 'exchange':
                    units = wallet['available']

            return units

    def get_cash(self, symbol='usd'):
        """."""

        return self.get_available_units(symbol=symbol)



    def get_symbols(self):

        return self.bitfinex.symbols()

    # def get_history(self, symbol, start, end):
    #     raise NotImplementedError("Not implemented yet")

    def get_pending_orders(self):

        return self.bitfinex.active_orders()

    def status_order(self, order_id):

        return self.bitfinex.status_order(order_id)

    def get_open_positions(self):

        ## todo test this

        positions = []


        balances = self.bitfinex.balances()
        for wallet in balances:
            if float(wallet['available']) > 0.0 and wallet['type'] == 'exchange':
                positions.append(wallet)

        return positions

    # def cancel_orders(self, instrument):
    #     return self.bitfinex.or

    # def modify_stop_loss_onPosition(self, trade_id, stopLoss):
    #     return self.oanda.modify_trade(self.OANDA_ACCOUNT_ID, trade_id, stopLoss = stopLoss)
    #
    # def modify_take_profit_onPosition(self, trade_id, takeProfit):
    #     return self.oanda.modify_trade(self.OANDA_ACCOUNT_ID, trade_id, takeProfit = takeProfit)
    #
    # def modify_trailing_stop_onPosition(self, trade_id, trailingStop):
    #     return self.oanda.modify_trade(self.OANDA_ACCOUNT_ID, trade_id, trailingStop = trailingStop)

    # def close_trade(self, trade_id):
    #     return self.oanda.close_trade(self.OANDA_ACCOUNT_ID, trade_id)

    # def get_open_positions(self, instrument = None):
    #     if instrument == None:
    #         return self.oanda.get_positions(self.OANDA_ACCOUNT_ID)
    #     else:
    #         return self.oanda.get_positions(self.OANDA_ACCOUNT_ID, instrument = instrument)
    #
