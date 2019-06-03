
import talib

import numpy as np
import pandas as pd

from htr.core.events import *
from htr.core.strategy import Strategy
from htr.helpers.ml import Classifier
from htr.helpers import Welford



class GaussianMix(Strategy):
    """
      .
       """

    def __init__(self, context, events, data_handler, slow_period=7, fast_period=3):
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

        self.symbol_list = self.data_handler.symbol_list
        self.events = events

        self.period = slow_period
        self.fast_period = fast_period
        self.minimum_data_size = self.period + 14
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
        self.hte = []
        self.autocov = []
        self.welford = Welford(self.fast_period)
        ## Add big sample
        price_data = pd.read_csv('C:\\Users\\utilizador\\Documents\\quant_research\\data\\basic_{}_sample.csv'.format(self.ticker.lower()))[
            ['Date', 'Close']]
        split = 100
        #split = 10000
        ## todo pre ml
        price_data = price_data[:split]
        price_data.columns = ['Date', self.ticker]
        price_data = price_data.drop(['Date'], axis=1)

        ## Update with 2k of recent 15m data
        recent_data = pd.read_csv('C:\\Users\\utilizador\\Documents\\quant_research\\data\\{}15.csv'.format(self.ticker), names=['Date','Time','Open', 'High', 'Low', 'Close', 'Volume' ])[['Time', 'Close']]

        recent_data.columns = ['Time', self.ticker]
        recent_data = recent_data.drop(['Time'], axis=1)
        ##
        price_data = recent_data[-100:].append(price_data, ignore_index=True)
        self.X = price_data.append(recent_data, ignore_index=True)

        # First training
        self.model = Classifier(self.ticker)
        print("Creating sample for {}.".format(self.ticker))
        self.model.update_sample(self.X.copy(), period=self.period)

        # ## TODO only for stoch pre ml testing
        self.X = self.X[-5:]

        # ## todo
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
            symbol, "Close", N=100
        )
        bar_date = self.data_handler.get_latest_bar_datetime(symbol)

        return bar_date, data



    def _check_stop(self, data):
        """Check for stop conditions, e.g. Stop losses, take profit, etc

        Generates EXIT signals.

        Returns:
            bool: True if EXIT signal was issued, False if it was not.
        """
        ## todo improve this
        symbol = self.symbol_list[0]
        if self.bought[symbol][0] != 'OUT':
            ret = (data[-1] - self.bought[symbol][1]) / self.bought[symbol][1] * 100
            if self.bought[symbol][0] == 'LONG':
                if ret < -0.06:
                    return True
            elif self.bought[symbol][0] == 'SHORT':
                if ret > 0.06:
                    return True
        return False

    def _check_week_stop(self, date):
        """Check for weekend.

        Generates EXIT signals.

        Returns:
            bool: True if EXIT signal was issued, False if it was not.
        """

        weekstop = False
        ## Friday
        if date.weekday() == 4 and date.hour >= 20:
            weekstop = True
        ## Saturday
        elif date.weekday() == 5:
            weekstop = True
        ## Sunday
        elif  date.weekday() == 6:
            weekstop = True

        return weekstop

    def _check_spread_stop(self, date):
        """Check for spread spikes.

        Generates EXIT signals.

        Returns:
            bool: True if EXIT signal was issued, False if it was not.
        """

        ## todo should actually monitor bid ask spread


        if date.hour >= 21 or date.hour < 1:
            return True
        return False

    def _fire_sale(self, bar_date, data):
        """Close all positions."""
        symbol = self.symbol_list[0]
        if self.bought[symbol][0] == 'SHORT':
            self.signal = 0.0
            print("CLOSE POSITION: %s" % bar_date)
            signal = SignalEvent(1, symbol, bar_date, 'EXIT', 1.0)
            self.bought[symbol] = ('OUT', data[-1])
            self.events.put(signal)

        elif self.bought[symbol][0] == 'LONG':
            self.signal = 0.0
            print("CLOSE POSITION: %s" % bar_date)
            signal = SignalEvent(1, symbol, bar_date, 'EXIT', 1.0)
            self.bought[symbol] = ('OUT', data[-1])
            self.events.put(signal)

    def hurst(self, p):
        tau = [];
        lagvec = []
        #  Step through the different lags
        for lag in range(2, 20):
            #  produce price difference with lag
            pp = np.subtract(p[lag:], p[:-lag])
            #  Write the different lags into a vector
            lagvec.append(lag)
            #  Calculate the variance of the difference vector
            tau.append(np.sqrt(np.std(pp)))
        #  linear fit to double-log graph (gives power)
        m = np.polyfit(np.log10(lagvec), np.log10(tau), 1)
        # calculate hurst
        hurst = m[0] * 2
        return hurst

    def lagged_auto_cov(self, Xi, t=1):
        """
        for series of values x_i, length N, compute empirical auto-cov with lag t
        defined: 1/(N-1) * \sum_{i=0}^{N-t} ( x_i - x_s ) * ( x_{i+t} - x_s )
        """
        N = len(Xi)

        # use sample mean estimate from whole series
        Xs = np.mean(Xi)

        # construct copies of series shifted relative to each other,
        # with mean subtracted from values
        end_padded_series = np.zeros(N + t)
        end_padded_series[:N] = Xi - Xs
        start_padded_series = np.zeros(N + t)
        start_padded_series[t:] = Xi - Xs
        try:
            auto_cov = 1. / (N - 1) * np.sum(start_padded_series * end_padded_series)
        except:
            auto_cov = 0.0
        return auto_cov

    def calculate_signals(self):
        """Calculates if trading signals should be generated and queued."""

        # For each symbol in the symbol list.
        for symbol in self.symbol_list:
            # Load price series data.
            bar_date, data = self._load_data(symbol)

            # Perform technical analysis.
            if data is not None and len(data) > self.minimum_data_size:

                # Checks for stop conditions
                if self._check_week_stop(bar_date) or self._check_stop(data):
                    self._fire_sale(bar_date, data)
                    return

                if self._check_spread_stop(bar_date):
                    return

                self.pred += 1
                vol = self.welford.update_and_return(data[-self.fast_period:]) * 100
                self.fprice = self.model.feature_gen.get_fprice()
                fret = sum(pd.DataFrame(self.fprice[-self.fast_period:]).pct_change()[-self.fast_period + 1:].values)
                # fsret = sum(pd.DataFrame(self.fprice[-self.period:]).pct_change()[-self.period + 1:].values)
                # sret = sum(pd.DataFrame(data[-self.period:]).pct_change()[-self.period + 1:].values)
                ret = sum(pd.DataFrame(data[-self.fast_period:]).pct_change()[-self.fast_period + 1:].values)
                print("\nRet: {}, Vol: {}".format(fret, vol))
                print("Iter: {}, row: {}".format(self.pred, [data[-1]]))
                self.X = self.X.append(pd.DataFrame([[data[-1]]], columns=[self.ticker]), ignore_index=True)
                self.X = self.X.iloc[1:]
                self.X.index = self.X.index.values + self.pred
                # Add new tick.
                # todo no ml version
                self.model.update_sample(self.X.copy(), fast=True, period=self.period)
                # self.model.update_sample(self.X.copy())
                # Generate features.

                # stoch_osc = self.model.feature_gen.get_stoch()

                # previous_stoch_osc = self.model.feature_gen.get_stoch()[-2]
                #previous_fprice = self.fprice
               # print('SIGNAL: {}'.format(self.signal))
               #  x_vec = self.model.get_x_vec()
               #  # print("x_vec: {}".format(x_vec))
               #  if self.fprice > previous_fprice:
               #      if self.y_pred == 1.0:
               #          self.hit += 1
               #      else:
               #          self.false_down += 1
               #          self.miss += 1
               #
               #  elif self.fprice < previous_fprice:
               #      if self.y_pred == -1.0:
               #          self.hit += 1
               #      else:
               #          self.miss += 1
               #          self.false_up += 1
               #
               #  self.y_pred = self.model.predict(x_vec)[0]
               #  print("\nReport: Hits: {}, Missed: {}, False_Up: {}, False_Down: {}.\n".format(self.hit, self.miss,
               #                                                                                 self.false_up,
               #                                                                                 self.false_down))
               #  print("Prediction: {}\n".format(self.y_pred))

                bbh = self.fprice[-1] + 3*vol / 100
                bbl = self.fprice[-1] - 3*vol / 100
                self.hte.append(pd.DataFrame(data[-21:]).rolling(21).apply(self.hurst).values[-1][0])
                # self.autocov.append(np.log(pd.DataFrame(data[-21:]).rolling(21).apply(self.lagged_auto_cov).values[-1][0]))
                self.autocov.append(
                    pd.DataFrame(data[-21:]).rolling(21).apply(lambda x: x.autocorr(), raw=False).values[-1][0])
                try:
                    if self.autocov[-1] > 50.0: self.autocov[-1] = self.autocov[-2]
                except:
                    pass
                # print("HTE {}".format(self.hte))
                reverting = False
                hurst_mean = np.mean(self.hte[-21:])
                hurst_std = np.std(self.hte[-21:])
                autocov_mean = np.mean(self.autocov)
                autocov_std = np.std(self.autocov)
                # if self.hte[-1] < -0.041 and self.hte[-1] > -0.79:
                # if self.hte[-1] < hurst_mean and self.hte[-1] >= hurst_mean - hurst_std:
                if self.autocov[-1] < autocov_mean:
                    reverting = True
                # reverting = True
                # print("Hurst: {} with mean {} is reverting? {}".format(self.hte[-1], np.mean(self.hte), reverting))
                ## todo trade only between 00h and 21h30, dont trade saturday,
                if data[-1] <= bbl and reverting and self.bought[symbol][0] == 'OUT':
                    self.signal = 1.0
                    print("LONG POSITION: %s" % bar_date)
                    signal = SignalEvent(1, symbol, bar_date, 'LONG', 1.0)
                    self.bought[symbol] = ('LONG', data[-1])
                    self.events.put(signal)

                elif self.fprice[-1] <= data[-1] and self.bought[symbol][0] == 'LONG':
                    self.signal = 0.0
                    print("CLOSE POSITION: %s" % bar_date)
                    signal = SignalEvent(1, symbol, bar_date, 'EXIT', 1.0)
                    self.bought[symbol] = ('OUT', data[-1])
                    self.events.put(signal)

                ####################  ----   ####################

                elif data[-1] >= bbh and reverting and self.bought[symbol][0] == 'OUT':
                    self.signal = -1.0
                    print("SHORT POSITION: %s" % bar_date)
                    signal = SignalEvent(1, symbol, bar_date, 'SHORT', 1.0)
                    self.bought[symbol] = ('SHORT', data[-1])
                    self.events.put(signal)

                elif self.fprice[-1] >= data[-1] and self.bought[symbol][0] == 'SHORT':
                    self.signal = 0.0
                    print("CLOSE POSITION: %s" % bar_date)
                    signal = SignalEvent(1, symbol, bar_date, 'EXIT', 1.0)
                    self.bought[symbol] = ('OUT', data[-1])
                    self.events.put(signal)


                # returns = pd.Series(data).pct_change()
                # sum_returns = sum(returns.values[-3:])
                # ret = (data[-1] - self.bought[symbol][1]) / self.bought[symbol][1]
                # if self.fprice == 0.0:
                #     self.fprice = data[-1]
                # #print("Price Pct Change: {}.".format(((data[-1] - data[-2]) / data[-2]) * 100))
                # print("FPrice Pct Change: {}.".format(((data[-1] - self.fprice) / self.fprice) * 100))
                # ## todo
                # #Train every 15minutes if hearbeat = 15s
                # if self.pred % 50 == 0 and self.pred > 1:
                #     print("Training {} forecasting model.".format(self.ticker))
                #     self.model.train()


