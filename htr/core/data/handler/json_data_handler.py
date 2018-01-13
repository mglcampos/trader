# import pandas_datareader.data as web
import datetime
import os
import os.path

import numpy as np
import pandas as pd

from htr.core.data.handler.data_handler import DataHandler
from htr.core.events.event import  MarketEvent
from htr.helpers.dataprep.dataprep import DataPrep


class JsonDataHandler(DataHandler):

    def __init__(self, context, events):
        self.events = events
        self.data_sources = context.data_sources
        self.symbol_list = [ds['symbol'] for ds in context.data_sources]
        self.prepare = context.data_preparation
        self.header = context.data_header
        self.dframes = {}
        self.latest_data = {}
        self.data_generator = {}
        self.symbol_data = {}
        ##todo should there be more dataprep modules? choose base on context?
        self.dataprep = DataPrep(context = context)
        self.continue_backtest = True
        self._load_csv_files()

    ##todo use decorator to validate types
    def _load_csv_files(self):
        """Loads comma delimited files."""

        path_list = None
        symbol = None
        comb_index = None

        for ds in self.data_sources:
            path_list = ds['path']
            symbol = ds['symbol']
            self.dframes[symbol] = []
            self.latest_data[symbol] = []
            # For each path in path list.
            for path in path_list:
                if path is not None and symbol is not None:
                    if os.path.isdir(path):
                        for root, dirs, files in os.walk(path):
                            for file in files:
                                if file.endswith('.csv') or file.endswith('.txt'):
                                    self.dframes[symbol].append(pd.read_csv(os.path.join(root, file),
                header=0, parse_dates=True, names=self.header))

                    elif os.path.isfile(path):
                        self.dframes[symbol].append(pd.read_csv(path,
                header=0, parse_dates=True, names=self.header))

                else:
                    ##todo raise custom exception
                    raise ValueError("Path or symbol_list is empty")

            # Prepare data.
            if self.prepare is True:
                self.dframes[symbol] = self.dataprep.prepare(self.dframes[symbol])

            # todo change this to use concatenation of dframes
            self.data_generator[symbol] = self.dframes[symbol][0]
            self.symbol_data[symbol] = self.data_generator[symbol]

            # Guarantee all data iterates over the same index.
            if comb_index is None:
                comb_index = self.data_generator[symbol].index

            else:
                comb_index.union(self.data_generator[symbol].index)

        for s in self.symbol_list:
            self.data_generator[s] = self.data_generator[s].reindex(index=comb_index, method='pad').iterrows()

    def get_symbol_data(self, symbol=None):

        try:
            return self.symbol_data[symbol]

        except KeyError:
            return self.symbol_data


    def get_start_date(self, symbol):
        # todo test this

        start_date = min([dt.index[0] for dt in self.dframes[symbol]])

        return start_date

    def _get_new_bar(self, symbol):

        try:
            for b in self.data_generator[symbol]:
                yield b
        except KeyError:
            print("That symbol is not available in the historical data set.")
            raise

    def get_latest_bar(self, symbol):

        try:
            rows = self.latest_data[symbol]
        except KeyError:
            print("That symbol is not available in the historical data set.")
            raise
        else:
            return rows[-1]

    def get_latest_bars(self, symbol, N=1):

        try:
            rows = self.latest_data[symbol]
        except KeyError:
            print("That symbol is not available in the historical data set.")
            raise
        else:
            return rows[-N:]

    def get_latest_bar_datetime(self, symbol):

        try:
            rows = self.latest_data[symbol]
        except KeyError:
            print("That symbol is not available in the historical data set.")
            raise
        else:
            return rows[-1][0]

    def get_latest_bar_value(self, symbol, column):


        try:
            rows = self.latest_data[symbol]
        except KeyError:
            print("That symbol is not available in the historical data set.")
            raise
        else:
            return getattr(rows[-1][1], column)

    def get_latest_bars_values(self, symbol, val_type, N=1):


        try:
            rows = self.get_latest_bars(symbol, N)            

        except KeyError:
            print("That symbol is not available in the historical data set.")
            raise

        else:
            try:
                return np.array([getattr(b[1], val_type) for b in rows])

            except KeyError:
                print('"{}" doesn\'t exist in header'.format(val_type))
                raise

    def update_bars(self):

        for s in self.symbol_list:
            try:
                bar = next(self._get_new_bar(s))
            except StopIteration:
                self.continue_backtest = False
            else:
                if bar is not None:
                    self.latest_data[s].append(bar)
        self.events.put(MarketEvent())



class SimpleHistoricDataHandler(DataHandler):
    """
    HistoricTXTDataHandler is designed to read CSV files for
    each requested symbol from disk and provide an interface
    to obtain the "latest" bar in a manner identical to a live
    trading interface from MT4 Historical Center
    """

    def __init__(self, symbol_list, start = datetime.datetime(2014, 1, 1), end=datetime.datetime.now(), events=None, csv_dir = None, s_file=None):
        """
        Initialises the historic data handler by requesting
        the location of the CSV files and a list of symbols.
        It will be assumed that all files are of the form
        'symbol.csv', where symbol is a string in the list.
        Parameters:
        events - The Event Queue.
        csv_dir - Absolute directory path to the CSV files.
        symbol_list - A list of symbol strings.
        """
        self.events = events
        self.csv_dir = csv_dir
        self.symbol_list = symbol_list
        self.data_generator = {}
        self.s_file = s_file
        self.forecasting_data = {}
        self.latest_data = {}
        self.continue_backtest = True

        self._open_convert_csv_files()

    def _open_convert_csv_files(self):
        """
        Opens the CSV files from the data directory, converting
        them into pandas DataFrames within a symbol dictionary.
        For this handler it will be assumed that the data is
        taken from Yahoo. Thus its format will be respected.
        """
        comb_index = None
        self.output_plot = {}
        for s in self.symbol_list:

            # Load the CSV file with no header information, indexed on date
            s_ = s.replace("/","")                          ##remove '/' from the string
            s_file = s_ + self.s_file

            ##TODO change this
            # filename = s_ + '_H1_2012'
            filename = self.s_file

            self.data_generator[s] = pd.io.parsers.read_csv(
                os.path.abspath('histdata/' + s_file),
                header=0, parse_dates=True,
                names=['Type', 'Day', 'Time', 'Open', 'High', 'Low', 'Close']
            )

            self.forecasting_data[s] = self.data_generator[s]
            # print(s_file)
            # print(self.data_generator[s])

            day = self.data_generator[s]['Day']
            minutes = self.data_generator[s]['Time']
            date_index = []
            for i in self.data_generator[s].index:

                date = str(day.ix[i]) + ' ' + str(minutes.ix[i])
                date = datetime.datetime.strptime(date, "%Y.%m.%d %H:%M")
                date_index.append(date)

            self.data_generator[s] = self.data_generator[s].set_index([date_index])
            print(self.data_generator[s])



            # Combine the index to pad forward values

            if comb_index is None:
                comb_index = self.data_generator[s].index
            else:
                comb_index.union(self.data_generator[s].index)
                # Set the latest symbol_data to None
                # self.latest_data[s] = []
                # Reindex the dataframes

            # self.output_plot[s] = []
            self.output_plot[s] = self.data_generator[s]

            self.data_generator[s] = self.data_generator[s]. \
                reindex(index=self.data_generator[s].index, method='pad').iterrows()

            self.latest_data[s] = []
        # for s in self.symbol_list:
        #     self.data_generator[s] = self.data_generator[s]. \
        #         reindex(index=comb_index, method='pad').iterrows()

    def get_symbol_data(self):
        return self.output_plot

    def _get_new_bar(self, symbol):
        """
        Returns the latest bar from the data feed.
        """
        for b in self.data_generator[symbol]:
            yield b

    def get_latest_bar(self, symbol):

        """
        Returns the last bar from the latest_symbol list.
        """
        try:
            bars_list = self.latest_data[symbol]
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
            bars_list = self.latest_data[symbol]
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
            bars_list = self.latest_data[symbol]
        except KeyError:
            print("That symbol is not available in the historical data set.")
            raise
        else:
            return bars_list[-1][0]

    def get_latest_bar_value(self, symbol, val_type):

        """
        Returns one of the Open, High, Low, Close, Volume or OI
        values from the pandas Bar series object.
        """
        try:
            bars_list = self.latest_data[symbol]
        except KeyError:
            print("That symbol is not available in the historical data set.")
            raise
        else:
            return getattr(bars_list[-1][1], val_type)

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
            return np.array([getattr(b[1], val_type) for b in bars_list])

    def get_latest_returns(self, symbol, val_type, N=3):
        """
        Returns the last N bar values from the
        latest_symbol list, or N-k if less available.
        """
        returns = self.forecasting_data[symbol][-N:].pct_change()

        return returns

    def update_bars(self):
        """
        Pushes the latest bar to the latest_data structure
        for all symbols in the symbol list.
        """

        for s in self.symbol_list:
            try:
                bar = next(self._get_new_bar(s))
            except StopIteration:
                self.continue_backtest = False
            else:
                if bar is not None:
                    self.latest_data[s].append(bar)
        self.events.put(MarketEvent())

    def normalize(self, c):
        return (c - pd.rolling_mean(c, window=20)) / pd.rolling_std(c, window=20)

    def create_lagged_series(self, symbol, start_date, end_date, lags=5):
        # symbol = 'EUR/USD'


        # df = self.data_generator[symbol]
        # df = pd.concat(list(pd.read_csv(Reader(self.data_generator[symbol]), chunksize=10000)), axis=1)
        # df = pd.read_csv(Reader(self.data_generator[symbol]))
        # print(self.data_generator[symbol])
        # df = pd.DataFrame.from_records(list(self.data_generator[symbol]))
        df = self.forecasting_data[symbol]
        df = df.reset_index()


        rm = pd.rolling_mean(df['Close'], window=20)


        # TODO move this indicators to toolbox

        df['Momentum'] = float('NaN')
        df['SMA'] = float('NaN')
        df['Bollinger'] = float('NaN')
        df['STD'] = pd.rolling_std(df['Close'], window=20)

        for i in range(0, df.shape[0]):
            # print(datetime.datetime(df.loc[i,'Day'].replace('.','-')))
            # df.loc[i,'Day'] = datetime.datetime(df.loc[i,'Day'].replace('.','-'))
            if i >= 5:
                df.loc[i, 'Momentum'] = (df.loc[i, 'Close'] / df.loc[i - 5, 'Close'] - 1) * 100
            df.loc[i, 'SMA'] = ((df.loc[i, 'Close'] / rm[i]) - 1) * 100
            df.loc[i, 'Bollinger'] = ((df.loc[i, 'Close'] - rm[i]) / (2 * df.loc[i, 'STD']))

        # print(rm[45], " Rolling mean")
        # print(df.loc[45, 'Close'], 'Price')
        # print(df.loc[45, 'SMA'], 'SMA')
        # print(df.loc[45, 'Close'] - df.loc[45, 'SMA'])
        # print(df.loc[45, 'Bollinger'], 'Bollinger')
        # print(df['Momentum'])
        # print(df['SMA'])
        # print(df['Bollinger'])

        # plt.show()
        df['Bollinger'] = self.normalize(df['Bollinger'])
        df['SMA'] = self.normalize(df['SMA'])
        df['Momentum'] = self.normalize(df['Momentum'])
        # df['Volume'] = self.normalize(df['Volume'])


        # Create the new lagged DataFrame
        tslag = pd.DataFrame(index=df.index)
        # tslag["Today"] = df["Adj Close"]
        tslag["Today"] = df["Close"]
        # tslag["Volume"] = df["Volume"]
        tslag["Day"] = df["Day"]

        # Create the shifted lag series of prior trading period close values
        for i in range(0, lags):
            # tslag["Lag%s" % str(i+1)] = df["Adj Close"].shift(i+1)
            tslag["Lag%s" % str(i + 1)] = df["Close"].shift(i + 1)
        # Create the returns DataFrame
        tsret = pd.DataFrame(index=tslag.index)
        # tsret["Volume"] = tslag["Volume"]
        tsret["Day"] = tslag["Day"]
        tsret["Momentum"] = df["Momentum"]
        tsret["Bollinger"] = df["Bollinger"]
        tsret["SMA"] = df["SMA"]
        tsret["Today"] = tslag["Today"].pct_change()*100.0
        # If any of the values of percentage returns equal zero, set them to
        # a small number (stops issues with QDA model in Scikit-Learn)
        for i,x in enumerate(tsret["Today"]):
            if (abs(x) < 0.0001):
                tsret["Today"][i] = 0.0001
        # Create the lagged percentage returns columns
        for i in range(0, lags):
            tsret["Lag%s" % str(i+1)] = \
            tslag["Lag%s" % str(i+1)].pct_change()*100.0
        # Create the "Direction" column (+1 or -1) indicating an up/down day
        tsret["Direction"] = np.sign(tsret["Today"])
        # print(tsret['Day'])
        for i in range(0, tsret.shape[0]):
            date = tsret.loc[i,'Day'].split('.')
            tsret.loc[i, 'Day'] = datetime.datetime(int(date[0]),int(date[1]),int(date[2]))
        tsret = tsret[tsret['Day'] >= start_date]

        return tsret[lags:-1]


class HistoricTXTDataHandler(DataHandler):
    """
    HistoricTXTDataHandler is designed to read CSV files for
    each requested symbol from disk and provide an interface
    to obtain the "latest" bar in a manner identical to a live
    trading interface from MT4 Historical Center
    """

    def __init__(self, symbol_list, start = datetime.datetime(2014, 1, 1), end=datetime.datetime.now(), events=None, csv_dir = None, s_file=None):
        """
        Initialises the historic data handler by requesting
        the location of the CSV files and a list of symbols.
        It will be assumed that all files are of the form
        'symbol.csv', where symbol is a string in the list.
        Parameters:
        events - The Event Queue.
        csv_dir - Absolute directory path to the CSV files.
        symbol_list - A list of symbol strings.
        """
        self.events = events
        self.csv_dir = csv_dir
        self.symbol_list = symbol_list
        self.data_generator = {}
        self.s_file = s_file
        self.forecasting_data = {}
        self.latest_data = {}
        self.continue_backtest = True

        self._open_convert_csv_files()

    def _open_convert_csv_files(self):
        """
        Opens the CSV files from the data directory, converting
        them into pandas DataFrames within a symbol dictionary.
        For this handler it will be assumed that the data is
        taken from Yahoo. Thus its format will be respected.
        """
        comb_index = None
        self.output_plot = {}
        for s in self.symbol_list:

            # Load the CSV file with no header information, indexed on date
            s_ = s.replace("/","")                          ##remove '/' from the string
            s_file = s_ + self.s_file
            # self.data_generator[s] = pd.io.parsers.read_csv(
            #     os.path.join(self.csv_dir, '% s.csv' % s_file),
            #     header=0, index_col=0, parse_dates=True, names=['Day','Time','Open','Max','Min','Close','Volume']
            # ).sort_index()

            # self.data_generator[s] = pd.io.parsers.read_csv(
            #     os.path.join(self.csv_dir + s_file + '/', '% s.txt' % s_file),
            #     header=0, parse_dates=True,
            #     names=['Type','Day', 'Time', 'Open', 'Max', 'Min', 'Close']
            # )
            self.data_generator[s] = pd.io.parsers.read_csv(
                os.path.join(self.csv_dir + s_ + '/', '% s' % s_file),
                header=0, parse_dates=True,
                names=['Type', 'Day', 'Time', 'Open', 'Max', 'Min', 'Close']
            )
            self.forecasting_data[s] = self.data_generator[s]
            print(s_file)
            # print(self.data_generator[s])

            day = self.data_generator[s]['Day']
            minutes = self.data_generator[s]['Time']
            date_index = []
            for i in self.data_generator[s].index:

                date = str(day.ix[i]) + ' ' + str(minutes.ix[i])
                date = datetime.datetime.strptime(date, "%Y.%m.%d %H:%M")
                date_index.append(date)

            self.data_generator[s] = self.data_generator[s].set_index([date_index])
            print(self.data_generator[s])



            # Combine the index to pad forward values

            if comb_index is None:
                comb_index = self.data_generator[s].index
            else:
                comb_index.union(self.data_generator[s].index)
                # Set the latest symbol_data to None
                # self.latest_data[s] = []
                # Reindex the dataframes

            # self.output_plot[s] = []
            self.output_plot[s] = self.data_generator[s]

            self.data_generator[s] = self.data_generator[s]. \
                reindex(index=self.data_generator[s].index, method='pad').iterrows()

            self.latest_data[s] = []
        # for s in self.symbol_list:
        #     self.data_generator[s] = self.data_generator[s]. \
        #         reindex(index=comb_index, method='pad').iterrows()

    def get_symbol_data(self):
        return self.output_plot

    def _get_new_bar(self, symbol):
        """
        Returns the latest bar from the data feed.
        """
        for b in self.data_generator[symbol]:
            yield b

    def get_latest_bar(self, symbol):

        """
        Returns the last bar from the latest_symbol list.
        """
        try:
            bars_list = self.latest_data[symbol]
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
            bars_list = self.latest_data[symbol]
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
            bars_list = self.latest_data[symbol]
        except KeyError:
            print("That symbol is not available in the historical data set.")
            raise
        else:
            return bars_list[-1][0]

    def get_latest_bar_value(self, symbol, val_type):

        """
        Returns one of the Open, High, Low, Close, Volume or OI
        values from the pandas Bar series object.
        """
        try:
            bars_list = self.latest_data[symbol]
        except KeyError:
            print("That symbol is not available in the historical data set.")
            raise
        else:
            return getattr(bars_list[-1][1], val_type)

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
            return np.array([getattr(b[1], val_type) for b in bars_list])

    def get_latest_returns(self, symbol, val_type, N=3):
        """
        Returns the last N bar values from the
        latest_symbol list, or N-k if less available.
        """
        returns = self.forecasting_data[symbol][-N:].pct_change()

        return returns

    def update_bars(self):
        """
        Pushes the latest bar to the latest_data structure
        for all symbols in the symbol list.
        """

        for s in self.symbol_list:
            try:
                bar = next(self._get_new_bar(s))
            except StopIteration:
                self.continue_backtest = False
            else:
                if bar is not None:
                    self.latest_data[s].append(bar)
        self.events.put(MarketEvent())

    def normalize(self, c):
        return (c - pd.rolling_mean(c, window=20)) / pd.rolling_std(c, window=20)

    def create_lagged_series(self, symbol, start_date, end_date, lags=5):
        symbol = 'EUR/USDH1'


        # df = self.data_generator[symbol]
        # df = pd.concat(list(pd.read_csv(Reader(self.data_generator[symbol]), chunksize=10000)), axis=1)
        # df = pd.read_csv(Reader(self.data_generator[symbol]))
        # print(self.data_generator[symbol])
        # df = pd.DataFrame.from_records(list(self.data_generator[symbol]))
        df = self.forecasting_data[symbol]
        df = df.reset_index()


        rm = pd.rolling_mean(df['Close'], window=20)


        # TODO move this indicators to toolbox

        df['Momentum'] = float('NaN')
        df['SMA'] = float('NaN')
        df['Bollinger'] = float('NaN')
        df['STD'] = pd.rolling_std(df['Close'], window=20)

        for i in range(0, df.shape[0]):
            # print(datetime.datetime(df.loc[i,'Day'].replace('.','-')))
            # df.loc[i,'Day'] = datetime.datetime(df.loc[i,'Day'].replace('.','-'))
            if i >= 5:
                df.loc[i, 'Momentum'] = (df.loc[i, 'Close'] / df.loc[i - 5, 'Close'] - 1) * 100
            df.loc[i, 'SMA'] = ((df.loc[i, 'Close'] / rm[i]) - 1) * 100
            df.loc[i, 'Bollinger'] = ((df.loc[i, 'Close'] - rm[i]) / (2 * df.loc[i, 'STD']))

        # print(rm[45], " Rolling mean")
        # print(df.loc[45, 'Close'], 'Price')
        # print(df.loc[45, 'SMA'], 'SMA')
        # print(df.loc[45, 'Close'] - df.loc[45, 'SMA'])
        # print(df.loc[45, 'Bollinger'], 'Bollinger')
        # print(df['Momentum'])
        # print(df['SMA'])
        # print(df['Bollinger'])

        # plt.show()
        df['Bollinger'] = self.normalize(df['Bollinger'])
        df['SMA'] = self.normalize(df['SMA'])
        df['Momentum'] = self.normalize(df['Momentum'])
        df['Volume'] = self.normalize(df['Volume'])


        # Create the new lagged DataFrame
        tslag = pd.DataFrame(index=df.index)
        # tslag["Today"] = df["Adj Close"]
        tslag["Today"] = df["Close"]
        tslag["Volume"] = df["Volume"]
        tslag["Day"] = df["Day"]

        # Create the shifted lag series of prior trading period close values
        for i in range(0, lags):
            # tslag["Lag%s" % str(i+1)] = df["Adj Close"].shift(i+1)
            tslag["Lag%s" % str(i + 1)] = df["Close"].shift(i + 1)
        # Create the returns DataFrame
        tsret = pd.DataFrame(index=tslag.index)
        tsret["Volume"] = tslag["Volume"]
        tsret["Day"] = tslag["Day"]
        tsret["Momentum"] = df["Momentum"]
        tsret["Bollinger"] = df["Bollinger"]
        tsret["SMA"] = df["SMA"]
        tsret["Today"] = tslag["Today"].pct_change()*100.0
        # If any of the values of percentage returns equal zero, set them to
        # a small number (stops issues with QDA model in Scikit-Learn)
        for i,x in enumerate(tsret["Today"]):
            if (abs(x) < 0.0001):
                tsret["Today"][i] = 0.0001
        # Create the lagged percentage returns columns
        for i in range(0, lags):
            tsret["Lag%s" % str(i+1)] = \
            tslag["Lag%s" % str(i+1)].pct_change()*100.0
        # Create the "Direction" column (+1 or -1) indicating an up/down day
        tsret["Direction"] = np.sign(tsret["Today"])
        # print(tsret['Day'])
        # for i in range(0, tsret.shape[0]):
        #     date = tsret.loc[i,'Day'].split('.')
        #     tsret.loc[i, 'Day'] = datetime.datetime(int(date[0]),int(date[1]),int(date[2]))
        tsret = tsret[tsret['Day'] >= start_date]

        return tsret[lags:-1]

class HistoricYahooCSVDataHandler(DataHandler):
    """
    HistoricCSVDataHandler is designed to read CSV files for
    each requested symbol from disk and provide an interface
    to obtain the "latest" bar in a manner identical to a live
    trading interface from Yahoo Finance
    """

    def __init__(self, symbol_list, start = datetime.datetime(2014, 1, 1), end=datetime.datetime.now(), events=None, csv_dir = None):
        """
        Initialises the historic data handler by requesting
        the location of the CSV files and a list of symbols.
        It will be assumed that all files are of the form
        'symbol.csv', where symbol is a string in the list.
        Parameters:
        events - The Event Queue.
        csv_dir - Absolute directory path to the CSV files.
        symbol_list - A list of symbol strings.
        """
        self.events = events
        self.csv_dir = csv_dir
        self.symbol_list = symbol_list
        self.data_generator = {}
        self.latest_data = {}
        self.continue_backtest = True
        self._open_convert_csv_files()


    def _open_convert_csv_files(self):
        """
        Opens the CSV files from the data directory, converting
        them into pandas DataFrames within a symbol dictionary.
        For this handler it will be assumed that the data is
        taken from Yahoo. Thus its format will be respected.
        """
        comb_index = None
        for s in self.symbol_list:

            # Load the CSV file with no header information, indexed on date
            s_file = s.replace("/","")                          ##remove '/' from the string
            self.data_generator[s] = pd.io.parsers.read_csv(
                os.path.join(self.csv_dir, '% s.csv' % s_file),
                header=0, index_col=0, parse_dates=True,
                names=[
                    'datetime', 'open', 'high',
                    'low', 'close', 'volume', 'Close'
                ]
            ).sort_index()
            # Combine the index to pad forward values

            if comb_index is None:
                comb_index = self.data_generator[s].index
            else:
                comb_index.union(self.data_generator[s].index)
                # Set the latest symbol_data to None
                # self.latest_data[s] = []
                # Reindex the dataframes
        self.latest_data[s] = []
        for s in self.symbol_list:
            self.data_generator[s] = self.data_generator[s]. \
                reindex(index=comb_index, method='pad').iterrows()

    def _get_new_bar(self, symbol):
        """
        Returns the latest bar from the data feed.
        """
        for b in self.data_generator[symbol]:
            yield b

    def get_latest_bar(self, symbol):

        """
        Returns the last bar from the latest_symbol list.
        """
        try:
            bars_list = self.latest_data[symbol]
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
            bars_list = self.latest_data[symbol]
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
            bars_list = self.latest_data[symbol]
        except KeyError:
            print("That symbol is not available in the historical data set.")
            raise
        else:
            return bars_list[-1][0]

    def get_latest_bar_value(self, symbol, val_type):

        """
        Returns one of the Open, High, Low, Close, Volume or OI
        values from the pandas Bar series object.
        """
        try:
            bars_list = self.latest_data[symbol]
        except KeyError:
            print("That symbol is not available in the historical data set.")
            raise
        else:
            return getattr(bars_list[-1][1], val_type)

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
            return np.array([getattr(b[1], val_type) for b in bars_list])

    def update_bars(self):
        """
        Pushes the latest bar to the latest_data structure
        for all symbols in the symbol list.
        """

        for s in self.symbol_list:
            try:
                bar = next(self._get_new_bar(s))
            except StopIteration:
                self.continue_backtest = False
            else:
                if bar is not None:
                    self.latest_data[s].append(bar)
        self.events.put(MarketEvent())


class HistoricMT4CSVDataHandler(DataHandler):
    """
    HistoricCSVDataHandler is designed to read CSV files for
    each requested symbol from disk and provide an interface
    to obtain the "latest" bar in a manner identical to a live
    trading interface from MT4 Historical Center
    """

    def __init__(self, symbol_list, start = datetime.datetime(2014, 1, 1), end=datetime.datetime.now(), events=None, csv_dir = None):
        """
        Initialises the historic data handler by requesting
        the location of the CSV files and a list of symbols.
        It will be assumed that all files are of the form
        'symbol.csv', where symbol is a string in the list.
        Parameters:
        events - The Event Queue.
        csv_dir - Absolute directory path to the CSV files.
        symbol_list - A list of symbol strings.
        """
        self.events = events
        self.csv_dir = csv_dir
        self.symbol_list = symbol_list
        self.data_generator = {}
        self.forecasting_data = {}
        self.latest_data = {}
        self.continue_backtest = True
        self._open_convert_csv_files()


    def _open_convert_csv_files(self):
        """
        Opens the CSV files from the data directory, converting
        them into pandas DataFrames within a symbol dictionary.
        For this handler it will be assumed that the data is
        taken from Yahoo. Thus its format will be respected.
        """
        comb_index = None
        self.output_plot = {}
        for s in self.symbol_list:

            # Load the CSV file with no header information, indexed on date
            s_file = s.replace("/","")                          ##remove '/' from the string
            # self.data_generator[s] = pd.io.parsers.read_csv(
            #     os.path.join(self.csv_dir, '% s.csv' % s_file),
            #     header=0, index_col=0, parse_dates=True, names=['Day','Time','Open','Max','Min','Close','Volume']
            # ).sort_index()

            self.data_generator[s] = pd.io.parsers.read_csv(
                os.path.join(self.csv_dir, '% s.csv' % s_file),
                header=0, parse_dates=True,
                names=['Day', 'Time', 'Open', 'Max', 'Min', 'Close', 'Volume']
            )
            self.forecasting_data[s] = self.data_generator[s]
            print(s_file)
            # print(self.data_generator[s])

            day = self.data_generator[s]['Day']
            minutes = self.data_generator[s]['Time']
            date_index = []
            for i in self.data_generator[s].index:

                date = str(day.ix[i]) + ' ' + str(minutes.ix[i])
                date = datetime.datetime.strptime(date, "%Y.%m.%d %H:%M")
                date_index.append(date)

            self.data_generator[s] = self.data_generator[s].set_index([date_index])
            print(self.data_generator[s])



            # Combine the index to pad forward values

            if comb_index is None:
                comb_index = self.data_generator[s].index
            else:
                comb_index.union(self.data_generator[s].index)
                # Set the latest symbol_data to None
                # self.latest_data[s] = []
                # Reindex the dataframes

            # self.output_plot[s] = []
            self.output_plot[s] = self.data_generator[s]

            self.data_generator[s] = self.data_generator[s]. \
                reindex(index=self.data_generator[s].index, method='pad').iterrows()

            self.latest_data[s] = []
        # for s in self.symbol_list:
        #     self.data_generator[s] = self.data_generator[s]. \
        #         reindex(index=comb_index, method='pad').iterrows()

    def get_symbol_data(self):
        return self.output_plot

    def _get_new_bar(self, symbol):
        """
        Returns the latest bar from the data feed.
        """
        for b in self.data_generator[symbol]:
            yield b

    def get_latest_bar(self, symbol):

        """
        Returns the last bar from the latest_symbol list.
        """
        try:
            bars_list = self.latest_data[symbol]
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
            bars_list = self.latest_data[symbol]
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
            bars_list = self.latest_data[symbol]
        except KeyError:
            print("That symbol is not available in the historical data set.")
            raise
        else:
            return bars_list[-1][0]

    def get_latest_bar_value(self, symbol, val_type):

        """
        Returns one of the Open, High, Low, Close, Volume or OI
        values from the pandas Bar series object.
        """
        try:
            bars_list = self.latest_data[symbol]
        except KeyError:
            print("That symbol is not available in the historical data set.")
            raise
        else:
            return getattr(bars_list[-1][1], val_type)

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
            return np.array([getattr(b[1], val_type) for b in bars_list])

    def get_latest_returns(self, symbol, val_type, N=3):
        """
        Returns the last N bar values from the
        latest_symbol list, or N-k if less available.
        """
        returns = self.forecasting_data[symbol][-N:].pct_change()

        return returns

    def update_bars(self):
        """
        Pushes the latest bar to the latest_data structure
        for all symbols in the symbol list.
        """

        for s in self.symbol_list:
            try:
                bar = next(self._get_new_bar(s))
            except StopIteration:
                self.continue_backtest = False
            else:
                if bar is not None:
                    self.latest_data[s].append(bar)
        self.events.put(MarketEvent())

    def normalize(self, c):
        return (c - pd.rolling_mean(c, window=20)) / pd.rolling_std(c, window=20)

    def create_lagged_series(self, symbol, start_date, end_date, lags=5):
        symbol = 'EUR/USDH1'


        # df = self.data_generator[symbol]
        # df = pd.concat(list(pd.read_csv(Reader(self.data_generator[symbol]), chunksize=10000)), axis=1)
        # df = pd.read_csv(Reader(self.data_generator[symbol]))
        # print(self.data_generator[symbol])
        # df = pd.DataFrame.from_records(list(self.data_generator[symbol]))
        df = self.forecasting_data[symbol]
        df = df.reset_index()


        rm = pd.rolling_mean(df['Close'], window=20)


        # TODO move this indicators to toolbox

        df['Momentum'] = float('NaN')
        df['SMA'] = float('NaN')
        df['Bollinger'] = float('NaN')
        df['STD'] = pd.rolling_std(df['Close'], window=20)

        for i in range(0, df.shape[0]):
            # print(datetime.datetime(df.loc[i,'Day'].replace('.','-')))
            # df.loc[i,'Day'] = datetime.datetime(df.loc[i,'Day'].replace('.','-'))
            if i >= 5:
                df.loc[i, 'Momentum'] = (df.loc[i, 'Close'] / df.loc[i - 5, 'Close'] - 1) * 100
            df.loc[i, 'SMA'] = ((df.loc[i, 'Close'] / rm[i]) - 1) * 100
            df.loc[i, 'Bollinger'] = ((df.loc[i, 'Close'] - rm[i]) / (2 * df.loc[i, 'STD']))

        # print(rm[45], " Rolling mean")
        # print(df.loc[45, 'Close'], 'Price')
        # print(df.loc[45, 'SMA'], 'SMA')
        # print(df.loc[45, 'Close'] - df.loc[45, 'SMA'])
        # print(df.loc[45, 'Bollinger'], 'Bollinger')
        # print(df['Momentum'])
        # print(df['SMA'])
        # print(df['Bollinger'])

        # plt.show()
        df['Bollinger'] = self.normalize(df['Bollinger'])
        df['SMA'] = self.normalize(df['SMA'])
        df['Momentum'] = self.normalize(df['Momentum'])
        df['Volume'] = self.normalize(df['Volume'])


        # Create the new lagged DataFrame
        tslag = pd.DataFrame(index=df.index)
        # tslag["Today"] = df["Adj Close"]
        tslag["Today"] = df["Close"]
        tslag["Volume"] = df["Volume"]
        tslag["Day"] = df["Day"]

        # Create the shifted lag series of prior trading period close values
        for i in range(0, lags):
            # tslag["Lag%s" % str(i+1)] = df["Adj Close"].shift(i+1)
            tslag["Lag%s" % str(i + 1)] = df["Close"].shift(i + 1)
        # Create the returns DataFrame
        tsret = pd.DataFrame(index=tslag.index)
        tsret["Volume"] = tslag["Volume"]
        tsret["Day"] = tslag["Day"]
        tsret["Momentum"] = df["Momentum"]
        tsret["Bollinger"] = df["Bollinger"]
        tsret["SMA"] = df["SMA"]
        tsret["Today"] = tslag["Today"].pct_change()*100.0
        # If any of the values of percentage returns equal zero, set them to
        # a small number (stops issues with QDA model in Scikit-Learn)
        for i,x in enumerate(tsret["Today"]):
            if (abs(x) < 0.0001):
                tsret["Today"][i] = 0.0001
        # Create the lagged percentage returns columns
        for i in range(0, lags):
            tsret["Lag%s" % str(i+1)] = \
            tslag["Lag%s" % str(i+1)].pct_change()*100.0
        # Create the "Direction" column (+1 or -1) indicating an up/down day
        tsret["Direction"] = np.sign(tsret["Today"])
        # print(tsret['Day'])
        # for i in range(0, tsret.shape[0]):
        #     date = tsret.loc[i,'Day'].split('.')
        #     tsret.loc[i, 'Day'] = datetime.datetime(int(date[0]),int(date[1]),int(date[2]))
        tsret = tsret[tsret['Day'] >= start_date]

        return tsret


# class HistoricPandasDataHandler:
#     """
#         Gathers historical data
#     """
#     def __init__(self, symbol_list, start = datetime.datetime(2014, 1, 1), end=datetime.datetime.now(), events=None):
#         """
#
#         """
#         self.events = events
#         self.start = start
#         self.end = end
#         self.symbol_list = symbol_list
#         self.data_generator = {}
#         self.latest_data = {}
#         self.continue_backtest = True
#
#         self.generate_data(start, end)
#
#
#     def pandas_dict(self):
#         """
#             Encodes symbols to be able to make requests via pandas
#         """
#         symbol_pandas = {}
#         symbol_pandas['USD/EUR'] = "EUR=X"
#         symbol_pandas['USD/JPY'] = "JPY=X"
#         symbol_pandas['USD/CAD'] = "CAD=X"
#         symbol_pandas['USD/CHF'] = "CHF=X"
#         symbol_pandas['USD/AUD'] = "AUD=X"
#         symbol_pandas['USD/NZD'] = "NZD=X"
#         symbol_pandas['USD/GBP'] = "GBP=X"
#         symbol_pandas['EUR/USD'] = "EUR=X"
#         symbol_pandas['JPY/USD'] = "JPY=X"
#         symbol_pandas['CAD/USD'] = "CAD=X"
#         symbol_pandas['CHF/USD'] = "CHF=X"
#         symbol_pandas['AUD/USD'] = "AUD=X"
#         symbol_pandas['NZD/USD'] = "NZD=X"
#         symbol_pandas['GBP/USD'] = "GBP=X"
#
#         return symbol_pandas
#
#     def generate_data(self, start, end):
#         """
#             Get Forex daily historical data from pandas. Most recent is always delayed 2 days.
#         """
#
#         symbols = self.pandas_dict()
#         for s in self.symbol_list:
#             self.latest_data[s] = []
#             if (s == "EUR/USD"):
#                 data = web.DataReader(symbols['EUR/USD'], 'yahoo', start, end)
#                 data['Open'] = 1 / data['Open']
#                 data['High'] = 1 / data['High']
#                 data['Low'] = 1 / data['Low']
#                 data['Close'] = 1 / data['Close']
#                 data['Adj Close'] = 1 / data['Adj Close']
#                 self.data_generator[s] = data
#
#             elif (s == "JPY/USD"):
#                 data = web.DataReader(symbols['JPY/USD'], 'yahoo', start, end)
#                 data['Open'] = 1 / data['Open']
#                 data['High'] = 1 / data['High']
#                 data['Low'] = 1 / data['Low']
#                 data['Close'] = 1 / data['Close']
#                 data['Adj Close'] = 1 / data['Adj Close']
#                 self.data_generator[s] = data
#
#             elif (s == "CAD/USD"):
#                 data = web.DataReader(symbols['CAD/USD'], 'yahoo', start, end)
#                 data['Open'] = 1 / data['Open']
#                 data['High'] = 1 / data['High']
#                 data['Low'] = 1 / data['Low']
#                 data['Close'] = 1 / data['Close']
#                 data['Adj Close'] = 1 / data['Adj Close']
#                 self.data_generator[s] = data
#
#             elif (s == "CHF/USD"):
#                 data = web.DataReader(symbols['CHF/USD'], 'yahoo', start, end)
#                 data['Open'] = 1 / data['Open']
#                 data['High'] = 1 / data['High']
#                 data['Low'] = 1 / data['Low']
#                 data['Close'] = 1 / data['Close']
#                 data['Adj Close'] = 1 / data['Adj Close']
#                 self.data_generator[s] = data
#
#             elif (s == "AUD/USD"):
#                 data = web.DataReader(symbols['AUD/USD'], 'yahoo', start, end)
#                 data['Open'] = 1 / data['Open']
#                 data['High'] = 1 / data['High']
#                 data['Low'] = 1 / data['Low']
#                 data['Close'] = 1 / data['Close']
#                 data['Adj Close'] = 1 / data['Adj Close']
#                 self.data_generator[s] = data
#
#             elif (s == "NZD/USD"):
#                 data = web.DataReader(symbols['NZD/USD'], 'yahoo', start, end)
#                 data['Open'] = 1 / data['Open']
#                 data['High'] = 1 / data['High']
#                 data['Low'] = 1 / data['Low']
#                 data['Close'] = 1 / data['Close']
#                 data['Adj Close'] = 1 / data['Adj Close']
#                 self.data_generator[s] = data
#
#             elif (s == "GBP/USD"):
#                 data = web.DataReader(symbols['GBP/USD'], 'yahoo', start, end)
#                 data['Open'] = 1 / data['Open']
#                 data['High'] = 1 / data['High']
#                 data['Low'] = 1 / data['Low']
#                 data['Close'] = 1 / data['Close']
#                 data['Adj Close'] = 1 / data['Adj Close']
#                 self.data_generator[s] = data
#
#             for t in symbols:
#                 if (s == t):
#                     data = web.DataReader(symbols[t], 'yahoo', start, end)
#                     self.data_generator[s] = data
#                     break
#
#             # TODO make bar compatible
#             #self.data_generator[s].index.name = None
#
#
#             # comb_index = None
#             # if comb_index is None:
#             #     comb_index = self.data_generator[s].index
#             # else:
#             #     comb_index.union(self.data_generator[s].index)
#             #
#             # self.data_generator[s] = self.data_generator[s]. \
#             #     reindex(index=comb_index, method='pad').iterrows()
#             # print self.data_generator[s]
#
#     def _get_new_bar(self, symbol):
#         """
#         Returns the latest bar from the data feed.
#         """
#
#         for b in self.data_generator[symbol]:
#             yield b
#
#     def get_latest_bar(self, symbol):
#
#         """
#         Returns the last bar from the latest_symbol list.
#         """
#         try:
#             bars_list = self.latest_data[symbol]
#         except KeyError:
#             print("That symbol is not available in the historical data set.")
#             raise
#         else:
#             return bars_list[-1]
#
#     def get_latest_bars(self, symbol, N=1):
#
#         """
#         Returns the last N bars from the latest_symbol list,
#         or N-k if less available.
#         """
#
#         try:
#             bars_list = self.latest_data[symbol]
#         except KeyError:
#             print("That symbol is not available in the historical data set.")
#             raise
#         else:
#             return bars_list[-N:]
#
#     def get_latest_bar_datetime(self, symbol):
#
#         """
#         Returns a Python datetime object for the last bar.
#         """
#         try:
#             bars_list = self.latest_data[symbol]
#         except KeyError:
#             print("That symbol is not available in the historical data set.")
#             raise
#         else:
#             return bars_list[-1][0]
#
#     def get_latest_bar_value(self, symbol, val_type):
#
#         """
#         Returns one of the Open, High, Low, Close, Volume or OI
#         values from the pandas Bar series object.
#         """
#         try:
#             bars_list = self.latest_data[symbol]
#         except KeyError:
#             print("That symbol is not available in the historical data set.")
#             raise
#         else:
#             return getattr(bars_list[-1][1], val_type)
#
#     def get_latest_bars_values(self, symbol, val_type, N=1):
#         """
#         Returns the last N bar values from the
#         latest_symbol list, or N-k if less available.
#         """
#         print symbol
#         print val_type
#         print self.get_latest_bars(symbol, N)
#         try:
#             bars_list = self.get_latest_bars(symbol, N)
#         except KeyError:
#             print("That symbol is not available in the historical data set.")
#             raise
#         else:
#             return np.array([getattr(b[1], val_type) for b in bars_list])
#
#     def update_bars(self):
#         """
#         Pushes the latest bar to the latest_data structure
#         for all symbols in the symbol list.
#         """
#
#         for s in self.symbol_list:
#             try:
#                 bar = next(self._get_new_bar(s))
#             except StopIteration:
#                 self.continue_backtest = False
#             else:
#                 if bar is not None:
#
#                     self.latest_data[s].append(bar)
#         self.events.put(MarketEvent())
#
#
#     def get_symbol_data(self, symbol=None):
#         if symbol != None:
#             return self.data_generator[symbol]
#         return self.data_generator
#


# start = datetime.datetime(2008, 1, 1)
# end = datetime.datetime.now()
#
# a = HistoricPandasDataHandler(['USD/GBP', 'USD/EUR'], start, end).get_symbol_data()
#
# print a
