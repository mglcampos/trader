
import datetime
import numpy as np
import pandas as pd
from .data_handler import DataHandler
import zmq_client
import json, ast
import dateutil.parser

class OandaLiveDataHandler(DataHandler):
    """

    """

    def __init__(self, symbol_list, start = datetime.datetime(2014, 1, 1), end=datetime.datetime.now(), events=None, csv_dir = None):
        """

        """

        self.events = events
        self.symbol_list = symbol_list
        self.symbol_data = {}
        self.latest_symbol_data = {}
        self.data_generator = {}
        self.continue_backtest = True

        ### Uncomment to use stored live stream ticks
        # try:
        #     self.artic = Artic()
        # except Exception as e:
        #     print(e)
        
        ### Uncomment to use oanda api directly in the livedatahandler
        # if(brokerage == 'Oanda'):
        #     self.broker = BrokerOanda().get_oanda()
        # else:
        #     raise ValueError("Brokerage Unknown")
        self.socket = None

        for s in self.symbol_list:
            self.latest_symbol_data[s] = []

        # self.update_symbol_data()

    def startListener(self):
        context = zmq_client.Context()
        self.socket = context.socket(zmq_client.REQ)
        # self.socket.setsockopt(zmq_client.py.SUBSCRIBE, '')
        self.socket.connect("tcp://127.0.0.1:5558")


    def update_symbol_data(self, granularity="D", start=datetime.datetime(2014, 01, 01), end=datetime.datetime(2016, 01, 01), alignmentTimezone='Europe/Lisbon'):
        # self.continue_backtest == True
        if self.socket is not None:
            for symbol in self.symbol_list:
                request = 'tick'
                message = "{0} {1}".format(request, symbol)
                print("send_message - ", message)
                self.socket.send(message)
                response = self.socket.recv()
                print "Received reply ", request, "[", response, "]"
            if not response == 'Nothing New':
                s, data = response.split(' ', 1)
                data = ast.literal_eval(data)
                if any(symbol in s for s in self.symbol_data):
                    self.symbol_data[symbol] = self.symbol_data[symbol].append(pd.DataFrame.from_dict({'ask': [data['ask']], 'bid': [data['bid']] , 'time' : [dateutil.parser.parse(data['time'])]}))

                else:
                    self.symbol_data[symbol] = pd.DataFrame.from_dict({'ask': [data['ask']], 'bid': [data['bid']] , 'time' : [dateutil.parser.parse(data['time'])]})

            # print self.symbol_data[symbol]
            # print symbol


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

        ## Get data directly from oanda api
        # print self.broker.get_history(instrument=s, granularity=granularity, start=start, end=end, alignmentTimezone=alignmentTimezone)
        ## Get data from mongodb
        # self.symbol_data[s] = self.artic.getData(s)

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
            return bars_list[-1]['time']  ##TODO [0]

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
            return getattr(bars_list[-1], val_type) ##TODO [1]

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
            return np.array([getattr(b, val_type) for b in bars_list])  ##TODO b[1]

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
# instruments = ['eur/USD']
# oanda = OandaLiveDataHandler(instruments)
#
# while True:
#     oanda.update_symbol_data()
#     print "BAR - " + str(oanda._get_new_bar('eur/USD'))

