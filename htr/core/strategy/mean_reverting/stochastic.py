
import talib
import statsmodels.tsa.stattools as ts
import numpy as np
import pandas as pd

from htr.core.events import *
from htr.core.strategy import Strategy
from htr.helpers.ml import Classifier



class Stochastic(Strategy):
    """
      .
       """

    def __init__(self, context, events, data_handler, short_window=100, long_window=200):
        """
        Initialises the Moving Averages Strategy.

        Args:
            context (Context): Runtime context (e.g. Strategies, initial_capital, timeframe, etc).
            events (Queue): Event queue object where signals are shared.
            data_handler (DataHandler) - The DataHandler object that provides bar information.
            short_window - The short moving average lookback.
            long_window - The long moving average lookback.
        """
        self.data_handler = data_handler
        self.minimum_data_size = 1
        self.symbol_list = self.data_handler.symbol_list
        self.events = events
        self.short_window = short_window
        self.long_window = long_window
        self.bought = self._calculate_initial_bought()

        self.pos_count = {}
        self.signal = 0.0
        for s in self.symbol_list:
            self.pos_count[s] = 0

        self.hurst_cache = []
        self.adf_cache = []
        self.pred = 0
        self.fprice = 0.0
        self.ticker = self.symbol_list[0]
        self.y_pred = 0
        self.false_down = 0
        self.hit = 0
        self.miss = 0
        self.false_up = 0
        ## Add big sample
        price_data = pd.read_csv('C:\\Users\\utilizador\\Documents\\quant_research\\data\\basic_{}_sample.csv'.format(self.ticker.lower()))[
            ['Date', 'Close']]
        split = 100
        #split = 10000
        price_data = price_data[:split]
        price_data.columns = ['Date', self.ticker]
        price_data = price_data.drop(['Date'], axis=1)

        ## Update with 2k of recent 15m data
        recent_data = pd.read_csv('C:\\Users\\utilizador\\Documents\\quant_research\\data\\{}15.csv'.format(self.ticker), names=['Date','Time','Open', 'High', 'Low', 'Close', 'Volume' ])[['Time', 'Close']]

        recent_data.columns = ['Time', self.ticker]
        recent_data = recent_data.drop(['Time'], axis=1)

        price_data = recent_data[-300:].append(price_data, ignore_index=True)
        self.X = price_data.append(recent_data, ignore_index=True)

        # First training
        self.model = Classifier(self.ticker)
        print("Creating sample for {}.".format(self.ticker))
        self.model.update_sample(self.X.copy())
        ## todo
        # print("Training {} forecasting model.".format(self.ticker))
        # self.model.train()

    def _calculate_initial_bought(self):
        """Cache where open positions are stored for strategy use."""

        bought = {}
        for s in self.symbol_list:
            # Stores position status and price when status changed.
            bought[s] = ('OUT', 0)

        return bought

    def _load_data(self, symbol):
        """Imports data using the DataHandler.

        Args:
            symbol (str): Security symbol

        Returns:

        """
        data = self.data_handler.get_latest_bars_values(
            symbol, "Close", N=self.short_window
        )
        bar_date = self.data_handler.get_latest_bar_datetime(symbol)

        return bar_date, data

    def _check_stop(self):
        """Check for stop conditions, e.g. Stop losses, take profit, etc

        Generates EXIT signals.

        Returns:
            bool: True if EXIT signal was issued, False if it was not.
        """

        pass

    def calculate_signals(self):
        """Calculates if trading signals should be generated and queued."""

        # For each symbol in the symbol list.
        for symbol in self.symbol_list:
            # Load price series data.
            bar_date, data = self._load_data(symbol)

            # Checks for stop conditions
            if self._check_stop():
                return

            # Perform technical analysis.
            if data is not None and len(data) > self.minimum_data_size:
                self.pred += 1
                print("Iter: {}, row: {}".format(self.pred, [data[-1]]))
                self.X = self.X.append(pd.DataFrame([[data[-1]]], columns=[self.ticker]), ignore_index=True)
                self.X = self.X.iloc[1:]
                self.X.index = self.X.index.values + self.pred
                # Add new tick.
                self.model.update_sample(self.X.copy(), fast=True)
                # Generate features.
                stoch_osc = self.model.feature_gen.get_stoch()[-1]
                previous_fprice = self.fprice
                self.fprice = self.model.feature_gen.get_fprice()[-1]
                print("\nStoch_Osc: {}, FPrice: {}".format(stoch_osc, self.fprice))

                if stoch_osc < 20 and self.fprice > data[-1]:
                    if self.bought[symbol][0] == 'OUT':
                        self.signal = 1.0
                        print("LONG POSITION: %s" % bar_date)
                        signal = SignalEvent(1, symbol, bar_date, 'LONG', 1.0)
                        self.bought[symbol] = ('LONG', data[-1])
                        self.events.put(signal)
                    elif self.bought[symbol][0] == 'SHORT':
                        self.signal = 0.0
                        print("CLOSE POSITION: %s" % bar_date)
                        signal = SignalEvent(1, symbol, bar_date, 'EXIT', 1.0)
                        self.bought[symbol] = ('OUT', data[-1])
                        self.events.put(signal)

                elif stoch_osc > 80 and self.fprice < data[-1]:
                    if self.bought[symbol][0] == 'OUT':
                        self.signal = -1.0
                        print("SHORT POSITION: %s" % bar_date)
                        signal = SignalEvent(1, symbol, bar_date, 'SHORT', 1.0)
                        self.bought[symbol] = ('SHORT', data[-1])
                        self.events.put(signal)
                    elif self.bought[symbol][0] == 'LONG':
                        self.signal = 0.0
                        print("CLOSE POSITION: %s" % bar_date)
                        signal = SignalEvent(1, symbol, bar_date, 'EXIT', 1.0)
                        self.bought[symbol] = ('OUT', data[-1])
                        self.events.put(signal)
                ## todo
                # print('SIGNAL: {}'.format(self.signal))
                # x_vec = self.model.get_x_vec()
                # # print("x_vec: {}".format(x_vec))
                # if self.fprice > previous_fprice:
                #     if self.y_pred == 1.0:
                #         self.hit += 1
                #     else:
                #         self.false_down += 1
                #         self.miss += 1
                #
                # elif self.fprice < previous_fprice:
                #     if self.y_pred == -1.0:
                #         self.hit += 1
                #     else:
                #         self.miss += 1
                #         self.false_up += 1
                #
                # self.y_pred = self.model.predict(x_vec)[0]
                # print("\nReport: Hits: {}, Missed: {}, False_Up: {}, False_Down: {}.\n".format(self.hit, self.miss,
                #                                                                                self.false_up,
                #                                                                                self.false_down))
                # print("Prediction: {}\n".format(self.y_pred))
                # returns = pd.Series(data).pct_change()
                # sum_returns = sum(returns.values[-3:])
                # ret = (data[-1] - self.bought[symbol][1]) / self.bought[symbol][1]
                if self.fprice == 0.0:
                    self.fprice = data[-1]
                #print("Price Pct Change: {}.".format(((data[-1] - data[-2]) / data[-2]) * 100))
                print("FPrice Pct Change: {}.".format(((data[-1] - self.fprice) / self.fprice) * 100))
                ## todo
                #Train every 15minutes if hearbeat = 15s
                # if self.pred % 50 == 0 and self.pred > 1:
                #     print("Training {} forecasting model.".format(self.ticker))
                #     self.model.train()


