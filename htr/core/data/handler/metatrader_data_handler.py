
from datetime import datetime as dt
import numpy as np
import pandas as pd
import zmq
import json, ast
import time

from htr.core.events import MarketEvent

class MetatraderDataHandler():
    """

    """

    def __init__(self, context, events):
        """

        """

        self.events = events
        self.symbol_list = context.symbol_list
        self.symbol_data = {}
        self.latest_symbol_data = {}
        self.data_generator = {}
        self.continue_backtest = True
        self.socket = None

        for s in self.symbol_list:
            self.latest_symbol_data[s] = []

        print('Starting Tick Listener')
        self.__startListener()

    def __startListener(self):
        context = zmq.Context()

        # Create PULL Socket
        self.pullSocket = context.socket(zmq.PULL)
        self.pullSocket.connect("tcp://localhost:5556")

    def __tick(self):
        while True:
            try:
                tick = str(self.pullSocket.recv())
                # print("Request: ", request)
                # time.sleep(1)
                print(tick)
                return tick

            except Exception as e:
                print(e)
                self.errors.append(e.__str__() + str(dt.now()))
                break

    def update_symbol_data(self):

        if self.socket is not None:
            print('update')
            response = self.__tick()
            print("Received reply ", "[", response, "]")

            if not response == 'Nothing New':
                s, bid, ask = str(response).split('|',2)
                spread = bid - ask

                self.symbol_data[s] = self.symbol_data[s].append(pd.DataFrame.from_dict({'ASK': [ask], 'BID': [bid], 'CLOSE': [bid - (spread/2)]  , 'TIME' : [dt.now()]}))
                ## If dataframe gets to big empty it.
                if len(self.symbol_data[s].index) > 100000:
                    self.symbol_data[s] = self.symbol_data[s][-1000]

                else:
                    self.symbol_data[s] = pd.DataFrame.from_dict({'ASK': [ask], 'BID': [bid], 'CLOSE': [bid - (spread/2)] , 'TIME' : [dt.now()]})
                ## todo rethink this
                self.update_bars()

# #################################################################
#
#         if self.socket is not None:
#             msg = self.socket.recv()
#             symbol, data = msg.split(' ', 1)
#             # Only processes desired messages
#             if not any(symbol in s for s in self.symbol_list):
#                 return
#             data = ast.literal_eval(data)
#             comb_index = None
#             if any(symbol in s for s in self.symbol_data):
#                 self.symbol_data[symbol] = self.symbol_data[symbol].append(pd.DataFrame.from_dict({'ask': [data.get('ask')], 'bid': [data.get('bid')] , 'time' : [dateutil.parser.parse(data.get('time'))]}))
#                 # if comb_index is None:
#                 #     comb_index = self.symbol_data[symbol].index
#                 # else:
#                 #     comb_index.union(self.symbol_data[symbol].index)
#                 # self.data_generator[symbol] = self.symbol_data[symbol]. \
#                 #     reindex(index=comb_index, method='pad').iterrows()
#                 # print self.symbol_data[symbol]
#             else:
#                 self.symbol_data[symbol] = pd.DataFrame.from_dict({'ask': [data.get('ask')], 'bid': [data.get('bid')] , 'time' : [dateutil.parser.parse(data.get('time'))]})
#                 # comb_index = self.symbol_data[symbol].index
#                 # self.data_generator[symbol] = self.symbol_data[symbol]. \
#                 #     reindex(index=comb_index, method='pad').iterrows()
#
#
#
#             print self.symbol_data[symbol]
#             print symbol



    def _get_new_bar(self, symbol):
        """
        Returns the latest bar from the data feed.
        """
        # for b in self.symbol_data[symbol]:
        #     yield b
        if any(symbol in s for s in self.symbol_data):
            return self.symbol_data[symbol].tail(1)
        else:
            return None

    def get_latest_bar(self, symbol):

        """
        Returns the last bar from the latest_symbol list.
        """
        try:
            bars_list = self.latest_symbol_data[symbol]
        except KeyError:
            print("That symbol is not available in the historical data set.")
            raise
        else:
            return bars_list[-1]

    def get_latest_bars(self, symbol, N=1):

        """
        Returns the last N bars from the latest_symbol list,
        or N-k if less available.
        """

        try:
            bars_list = self.latest_symbol_data[symbol]
        except KeyError:
            print("That symbol is not available in the historical data set.")
            raise
        else:
            return bars_list[-N:]

    def get_latest_bar_datetime(self, symbol):

        """
        Returns a Python datetime object for the last bar.
        """
        try:
            bars_list = self.latest_symbol_data[symbol]
        except KeyError:
            print("That symbol is not available in the historical data set.")
            raise
        else:
            # print ("timeee",bars_list[-1]['time'])
            return bars_list[-1]['time'.upper()]  ##TODO [0]

    def get_latest_bar_value(self, symbol, val_type):

        """
        Returns one of the Open, High, Low, Close, Volume or OI
        values from the pandas Bar series object.
        """
        try:
            bars_list = self.latest_symbol_data[symbol]
        except KeyError:
            print("That symbol is not available in the historical data set.")
            raise
        else:
            return float(getattr(bars_list[-1], val_type.upper())) ##TODO [1]

    def get_latest_bars_values(self, symbol, val_type, N=1):
        """
        Returns the last N bar values from the
        latest_symbol list, or N-k if less available.
        """

        try:
            bars_list = self.get_latest_bars(symbol, N)
        except KeyError:
            print("That symbol is not available in the historical data set.")
            raise
        else:
            return np.array([float(getattr(b, val_type.upper())) for b in bars_list])  ##TODO b[1]

    def update_bars(self):
        """
        Pushes the latest bar to the latest_symbol_data structure
        for all symbols in the symbol list.
        """

        for s in self.symbol_list:
            # try:
            #     bar = next(self._get_new_bar(s))
            # except StopIteration:
            #     self.continue_backtest = False
            # else:
            #     if bar is not None:
            #         self.latest_symbol_data[s].append(bar)
            bar = self._get_new_bar(s)
            if bar is not None:
                self.latest_symbol_data[s].append(bar)

        self.events.put(MarketEvent())



#
# from htr.core.configuration import Context
# import time
# context = Context()
# setattr(context, 'symbol_list', ['XRPUSD'])
# import queue
# handler = CryptoLiveDataHandler(context, queue)
#
# while True:
#     time.sleep(1)
#     handler.update_symbol_data()
#     print("BAR - " + str(handler._get_new_bar('XRPUSD')))

# symbol = ('XRPUSD')
# context = zmq.py.Context()
# socket = context.socket(zmq.py.REQ)
# socket.connect("tcp://127.0.0.1:5558")
# request = 'tick'
# message = "{0} {1}".format(request, symbol)
# print("send_message - ", message)
# socket.send_string(message)
# response = socket.recv()
# print("Received reply ", request, "[", response, "]")