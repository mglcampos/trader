import datetime
import OandaWrapper
from OandaWrapper import Streamer

class BrokerOanda():


    def __init__(self):
        # print "------ System online -------", datetime.now()
        self.OANDA_ACCESS_TOKEN = "970b08eac223cc0fb6b9d50aa9bd7bd5-aa2497c02bec0264cc32c98909be36fd"
        self.OANDA_ACCOUNT_ID = 6068438
        self.oanda = OandaWrapper.API(environment="practice", access_token=self.OANDA_ACCESS_TOKEN)
        self.accounts = self.oanda.get_accounts()
        # print oanda.get_accounts()
        start = datetime.datetime(2014, 1, 1)
        end = datetime.datetime(2016, 1, 27)
        # print self.oanda.get_history(instrument="EUR_USD", granularity="M1")

        # response = oanda.get_prices(instruments="EUR_USD")
        # prices = response.get("prices")
        # buy_price = prices[0].get("bid")
        #
        # print "Buy at", buy_price

        # trade_id = oanda.create_order(OANDA_ACCOUNT_ID, instrument="EUR_USD",
        #                               units=1000,
        #                               side='buy',
        #                               type='market')
        #
        # print "Trading id", trade_id




    def streamRates(self, instruments):
        self.stream = Streamer()
        self.stream.start(instruments = instruments)

    def get_capital(self):
        response = self.oanda.get_account(self.OANDA_ACCOUNT_ID)
        capital = response.get('balance')   ##TODO balance vs marginAvail
        return capital


    def get_accounts(self):
        return self.accounts

    def get_instruments(self):
        return self.oanda.get_instruments(self.OANDA_ACCOUNT_ID)

    def get_oanda(self):
        return self.oanda

    def get_prices(self, symbol_list):
        for i in range(len(symbol_list)):
            symbol_list[i] = symbol_list[i].replace('/', '_', 1)

        return self.oanda.get_prices(instruments = symbol_list)   ##TODO  da para adicionar um since, erro no datetime

    def get_history(self, symbol, start, end):                    ##TODO  da pra retornar candlesticks
        raise NotImplementedError("Not implemented yet")

    def get_pending_orders(self, symbol=None):                    ##TODO da pra usar contagem, maxid e ids
        if symbol == None:
            return self.oanda.get_orders(self.OANDA_ACCOUNT_ID)
        else:
            return self.oanda.get_orders(self.OANDA_ACCOUNT_ID, instrument = symbol)

    """

     Parameters:
     symbol - instrument to order
     units - number of units to open order for
     side - buy or sell
     type - limit, stop, marketIfTouched or market
     expiry - limit, stop or marketIfTouched expiration in UTC
     price - limit, stop or marketIfTouched trigger price
     lowerBound  - the minimum execution price
     upperBound - the maximum execution price
     stopLoss - the stop loss price
     takeProfit - the take profit price
     trailingStop - the trailing stop distance in pips (up to one decimal place)
    """

    def create_short_order(self, symbol, units, lowerBound=None, upperBound=None, takeProfit=None, trailingStop=None, triggerPrice=None, stopLoss=None, expiry=None):
        symbol = symbol.replace('/', '_', 1)
        return self.oanda.create_order(self.OANDA_ACCOUNT_ID, instrument = symbol, units = units, lowerBound=lowerBound, upperBound=upperBound, takeProfit=takeProfit, trailingStop=trailingStop, triggerPrice=triggerPrice, stopLoss=stopLoss, expiry=expiry,  side = "sell", type = "market")


    def create_long_order(self, symbol, units, lowerBound=None, upperBound=None, takeProfit=None, trailingStop=None, triggerPrice=None, stopLoss=None, expiry=None):
        symbol = symbol.replace('/', '_', 1)

        return self.oanda.create_order(self.OANDA_ACCOUNT_ID, instrument=symbol, units=units,
                                             lowerBound=lowerBound, upperBound=upperBound, takeProfit=takeProfit,
                                             trailingStop=trailingStop, triggerPrice=triggerPrice, stopLoss=stopLoss,
                                             expiry=expiry, side="buy", type="market")

    def create_order(self, symbol, units,direction, lowerBound=None, upperBound=None, takeProfit=None, trailingStop=None, triggerPrice=None, stopLoss=None, expiry=None):
        symbol = symbol.replace('/', '_', 1)

        return self.oanda.create_order(self.OANDA_ACCOUNT_ID, instrument=symbol, units=units,
                                             lowerBound=lowerBound, upperBound=upperBound, takeProfit=takeProfit,
                                             trailingStop=trailingStop, triggerPrice=triggerPrice, stopLoss=stopLoss,
                                             expiry=expiry, side=direction, type="market")


    # TODO testar com todas as variaveis

    def modify_stop_loss_onOrder(self, order_id, price):
        return self.oanda.modify_order(self.OANDA_ACCOUNT_ID, order_id, stoploss = price)
        ##TODO acabar e testar

    def close_order(self, order_id):
        return self.oanda.close_order(self.OANDA_ACCOUNT_ID, order_id)

    def get_order_info(self, order_id):
        return self.oanda.get_order(self.OANDA_ACCOUNT_ID, order_id)

    def get_trades(self, trade_ids = None, instrument=None ):
        if instrument is not None and trade_ids is not None:
            return self.oanda.get_trades(self.OANDA_ACCOUNT_ID, instrument = instrument.replace('/','_',1), ids = trade_ids)
        elif instrument is not None:
            return self.oanda.get_trades(self.OANDA_ACCOUNT_ID, instrument = instrument.replace('/','_',1))
        elif trade_ids is not None:
            return self.oanda.get_trades(self.OANDA_ACCOUNT_ID, ids= trade_ids)
        else:
            return self.oanda.get_trades(self.OANDA_ACCOUNT_ID)

    def modify_stop_loss_onPosition(self, trade_id, stopLoss):
        return self.oanda.modify_trade(self.OANDA_ACCOUNT_ID, trade_id, stopLoss = stopLoss)

    def modify_take_profit_onPosition(self, trade_id, takeProfit):
        return self.oanda.modify_trade(self.OANDA_ACCOUNT_ID, trade_id, takeProfit = takeProfit)

    def modify_trailing_stop_onPosition(self, trade_id, trailingStop):
        return self.oanda.modify_trade(self.OANDA_ACCOUNT_ID, trade_id, trailingStop = trailingStop)

    def close_trade(self, trade_id):
        return self.oanda.close_trade(self.OANDA_ACCOUNT_ID, trade_id)

    def get_open_positions(self, instrument = None):
        if instrument == None:
            return self.oanda.get_positions(self.OANDA_ACCOUNT_ID)
        else:
            return self.oanda.get_positions(self.OANDA_ACCOUNT_ID, instrument = instrument)

    def close_all_positions(self, instrument):
        return self.oanda.close_position(self.OANDA_ACCOUNT_ID, instrument)

class BrokerIB:

    def __init__(self):
        raise NotImplementedError()



# broker = BrokerOanda()
# symbol = 'EUR/USD'
# # #
# # print broker.get_capital()
# print broker.get_prices(['EUR/USD'])
# # # print broker.get_instruments()
# # # print broker.get_pending_orders()
# #
# #Open position
# response = broker.get_prices([symbol])
# prices = response.get('prices')
# buy_price = prices[0].get('bid')
# print "Buying at - ", buy_price
# response_order = broker.create_short_order(symbol, 1000)
# print("Response_order - ", response_order)
# trade = response_order.get('tradeOpened')
# trade_id = trade.get('id')
# print "Trading id - ", trade_id
#
# print 'Trade - ', broker.get_trades(trade_ids = [trade_id])
#
# # Close position
# response_close = broker.close_trade(trade_id)
# print 'Response_close - ', response_close
# profit = response_close.get('profit')
# print "Profit - ", profit
#
# print 'Trades - ', broker.get_trades()

###close all trades for a instrument
# #
# broker = BrokerOanda()
# symbol = 'EUR/USD'
# open_trades = broker.get_trades(instrument=symbol.replace('/','_',1))
# trades = open_trades.get('trades')
# print("Trades - ",len(trades), trades)
# for i in range(0, len(trades)):
#     trade_id = trades[i].get('id')
#     trade_closed = broker.close_trade(trade_id)
#     print("Trade Closed - ", trade_closed)