# import pandas_datareader.data as web
import os
import os.path

import numpy as np
import pandas as pd

from htr.core.data.handler.data_handler import DataHandler
from htr.core.events.event import  MarketEvent
from htr.helpers.dataprep.dataprep import DataPrep


class CsvDataHandler(DataHandler):

    def __init__(self, context, events):
        self.events = events
        self.data_sources = context.data_sources
        self.symbol_list = [ds['symbol'] for ds in context.data_sources]
        self.prepare = context.data_preparation
        self.header = context.data_header
        self.dframes = {}
        self.start_date = context.start_date
        self.end_date = context.end_date
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
                header=None, parse_dates=True, names=self.header))

                    elif os.path.isfile(path):
                        self.dframes[symbol].append(pd.read_csv(path,
                header=None, parse_dates=True, names=self.header))

                else:
                    ##todo raise custom exception
                    raise ValueError("Path or symbol_list is empty")

            # Prepare data.
            if self.prepare is True:
                self.dframes[symbol] = self.dataprep.prepare(self.dframes[symbol])
                ##todo change this later
                df = self.dframes[symbol][0]
                self.dframes[symbol][0] = df[(df.index > self.start_date) & (df.index < self.end_date)]

            # todo change this to use concatenation of dframes
            try:
                self.data_generator[symbol] = self.dframes[symbol][0]
            except:
                raise ValueError('No datasets were loaded, path may be wrong.')

            self.symbol_data[symbol] = self.data_generator[symbol]

            # Guarantee all data iterates over the same index.
            if comb_index is None:
                comb_index = self.data_generator[symbol].index

            else:
                comb_index.union(self.data_generator[symbol].index)
            print(self.data_generator[symbol])
        for s in self.symbol_list:
            self.data_generator[s] = self.data_generator[s].reindex(index=comb_index, method='pad').iterrows()

    def get_symbol_data(self, symbol=None):

        try:
            self.symbol_data[symbol] = self.symbol_data[symbol].dropna()
            return self.symbol_data[symbol]

        except KeyError:
            return self.symbol_data

    def get_start_date(self, symbol):
        # todo test this

        start_date = min([dt.index[0] for dt in self.dframes[symbol]])
        # start_date = self.symbol_data[symbol]['Date'][0]

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

