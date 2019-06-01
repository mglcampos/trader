# import pandas_datareader.data as web
import os
import os.path
from datetime import datetime as dt
import numpy as np
import pandas as pd
from influxdb import InfluxDBClient

from htr.core.data.handler.data_handler import DataHandler
from htr.core.events.event import  MarketEvent
from htr.helpers.dataprep.dataprep import DataPrep


class InfluxdbDataHandler(DataHandler):

    def __init__(self, context, events):
        self.events = events
        self.data_sources = context.data_sources
        self.symbol_list = [ds['symbol'] for ds in context.data_sources]
        self.ticker = self.symbol_list[0]
        self.prepare = context.data_preparation
        self.header = context.data_header
        self.freq = context.timeframe
        self.dframes = {}
        self.start_date = context.start_date
        self.end_date = context.end_date
        self.latest_data = {}
        self.latest_data[self.ticker] = []
        self.data_generator = {}
        self.symbol_data = {}
        self.db = InfluxDBClient('104.248.41.39', 8086, 'admin', 'jndm4jr5jndm4jr6', 'darwinex')

        ##todo should there be more dataprep modules? choose base on context?
        self.dataprep = DataPrep(context = context)
        self.continue_backtest = True
        try:
            self._load_data()
        except Exception as e:
            print('Error loading files.', e)

    def influx_to_pandas(self, result):
        df = pd.DataFrame(result, )
        df.index = pd.to_datetime(df['time'])

        return df.drop(['time'], axis=1)



    ##todo use decorator to validate types
    def _load_data(self):
        """Loads data."""

        start_dt = dt.strptime(self.start_date, '%Y-%m-%d')
        end_dt = dt.strptime(self.end_date, '%Y-%m-%d')
        start_epoch = int(float(start_dt.timestamp())) * 1000 * 1000 * 1000
        end_epoch = int(end_dt.timestamp()) * 1000 * 1000 * 1000
        query = "Select last(price) from {} where time > {} and time < {} group by time({})".format(self.ticker, str(
            start_epoch), str(end_epoch), self.freq)

        result = list(self.db.query(query))[0]
        data = self.influx_to_pandas(result)
        data = data.fillna(method='ffill')

        data.index.name = 'Datetime'
        data.columns = ['Close']
        self.dframes[self.ticker] = [data]
        self.data_generator[self.ticker] = self.dframes[self.ticker][0]
        self.symbol_data[self.ticker] = self.data_generator[self.ticker]
        self.data_generator[self.ticker] = self.data_generator[self.ticker].iterrows()

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

