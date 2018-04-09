from __future__ import print_function

from datetime import datetime

import statsmodels.api as sm
from sklearn.qda import QDA
from talib.abstract import *

try:
    import Queue as queue
except ImportError:
    import queue

import numpy as np
import pandas as pd
from htr.core.events.event import SignalEvent
from htr.core.strategy.strategy import Strategy



class MovingAverageCrossStrategy(Strategy):
    """
    Carries out a basic Moving Average Crossover strategy with a
    short/long simple weighted moving average. Default short/long
    windows are 100/400 periods respectively.
    """
    def __init__(self, bars, events, short_window=20, long_window=100, simulation=True):
        """
        Initialises the Moving Average Cross Strategy.
        Parameters:
        bars - The DataHandler object that provides bar information
        events - The Event Queue object.
        short_window - The short moving average lookback.
        long_window - The long moving average lookback.
        """
        self.bars = bars
        self.symbol_list = self.bars.symbol_list
        self.events = events
        self.short_window = short_window
        self.long_window = long_window
        # Set to True if a symbol is in the market
        self.bought = self._calculate_initial_bought()
        self.simulation = simulation
    def _calculate_initial_bought(self):


        bought = {}
        for s in self.symbol_list:
            bought[s] = 'OUT'
        return bought

    def _load_data(self, symbol):
        """
        Import data
        Args:
        	symbol:

        Returns:

        """
        bars = self.bars.get_latest_bars_values(
            symbol, "Close", N=self.long_window
        )
        bar_date = self.bars.get_latest_bar_datetime(symbol)

        return bar_date, bars


    def _check_stop(self):
        """Check for stop conditions, e.g. Stop losses, take profit, etc
            generates signals
            return true or false
        """
        pass


    def calculate_signals(self, event):


        if event.type == 'MARKET':

            for s in self.symbol_list:
                bar_date, bars = self._load_data()
                if self._check_stop():
                    return

                if bars is not None:
                    short_sma = np.mean(bars[-self.short_window:])
                    long_sma = np.mean(bars[-self.long_window:])
                    symbol = s
                    dt = bar_date
                    sig_dir = ""
                    print("self_bought",self.bought[s],s)
                    if short_sma > long_sma and self.bought[s] == "OUT":
                        print("LONG: %s" % bar_date)
                        signal = SignalEvent(1, symbol, dt, 'LONG', 1.0)
                        self.bought[s] = 'LONG'
                        self.events.put(signal)


                    elif short_sma < long_sma and self.bought[s] == "LONG":
                        print("CLOSE POSITION: %s" % bar_date)
                        signal = SignalEvent(1, symbol, dt, 'EXIT', 1.0)
                        self.bought[s] = 'OUT'
                        self.events.put(signal)
                    # if short_sma > long_sma and self.bought[s] == "OUT":
                    #     print("SHORT: %s" % bar_date)
                    #     signal = SignalEvent(1, symbol, dt, 'SHORT', 1.0)
                    #     self.bought[s] = 'SHORT'
                    #     self.events.put(signal)
                    #
                    #
                    # elif short_sma < long_sma and self.bought[s] == "SHORT":
                    #     print("CLOSE POSITION: %s" % bar_date)
                    #     signal = SignalEvent(1, symbol, dt, 'EXIT', 1.0)
                    #     self.bought[s] = 'OUT'
                    #     self.events.put(signal)

class MOMOStrategy(Strategy):
    """
    Momentum strategy with EMA and MACD
    """
    def __init__(self, bars, events, short_window=20, long_window=100, simulation=True):
        """
        Initialises the Moving Average Cross Strategy.
        Parameters:
        bars - The DataHandler object that provides bar information
        events - The Event Queue object.
        short_window - The short moving average lookback.
        long_window - The long moving average lookback.
        """
        self.bars = bars
        self.symbol_list = self.bars.symbol_list
        self.events = events
        self.short_window = short_window
        self.long_window = long_window
        # Set to True if a symbol is in the market
        self.bought = self._calculate_initial_bought()
        self.simulation = simulation
        self.daycounter = 0
    def _calculate_initial_bought(self):

        bought = {}
        for s in self.symbol_list:
            bought[s] = ('OUT',0)
        return bought

    def calculate_signals(self, event):


        if event.type == 'MARKET':
            for s in self.symbol_list:
                bars = self.bars.get_latest_bars_values(
                    s, "Close", N=27
                )
                day = self.bars.get_latest_bars_values(
                    s, "Day", N=27
                )
                high = self.bars.get_latest_bars_values(
                    s, "High", N=27
                )
                time = self.bars.get_latest_bars_values(
                    s, "Time", N=27
                )
                low = self.bars.get_latest_bars_values(
                    s, "Low", N=27
                )


                bar_date = self.bars.get_latest_bar_datetime(s)
                if bars is not None:

                    weekday = datetime(int(float(day[-1].split('.')[0])),int(float(day[-1].split('.')[1])),int(float(day[-1].split('.')[2])),int(float(time[-1].split(':')[0])),int(float(time[-1].split(':')[1]))).weekday()
                    print('weekday',weekday)
                    short_sma = np.mean(bars[-self.short_window:])
                    long_sma = np.mean(bars[-self.long_window:])
                    symbol = s
                    dt = bar_date
                    sig_dir = ""
                    print("self_bought",self.bought[s],s)

                    ema26 = pd.stats.moments.ewma(bars, 26)
                    ema12 = pd.stats.moments.ewma(bars, 12)
                    ema20 = pd.stats.moments.ewma(bars, 20)
                    macd = (ema12 - ema26) * 100

                    if (float(weekday) == 4 or bars[-1] + 0.002 < self.bought[s][1]) and self.bought[s][0] == "LONG":
                        print("CLOSE POSITION: %s" % bar_date)
                        signal = SignalEvent(1, symbol, dt, 'EXIT', 1.0)
                        self.bought[s] = ('OUT', 0)
                        self.events.put(signal)

                    elif (float(weekday) == 4 or bars[-1] - 0.002 > self.bought[s][1]) and self.bought[s][0] == "SHORT":
                        print("CLOSE POSITION: %s" % bar_date)
                        signal = SignalEvent(1, symbol, dt, 'EXIT', 1.0)
                        self.bought[s] = ('OUT', 0)
                        self.events.put(signal)

                    if bars[-1] > ema20[-1] and macd[-1] > -0.01 and self.bought[s][0] == "OUT":
                        print("LONG: %s" % bar_date)
                        signal = SignalEvent(1, symbol, dt, 'LONG', 1.0)
                        self.bought[s] = ('LONG',bars[-1])
                        self.events.put(signal)


                    elif bars[-1] < ema20[-1] and macd[-1] < 0.01 and self.bought[s][0] == "OUT":
                        print("SHORT: %s" % bar_date)
                        signal = SignalEvent(1, symbol, dt, 'SHORT', 1.0)
                        self.bought[s] = ('SHORT', bars[-1])
                        self.events.put(signal)


                    elif bars[-1] - 0.001 >= ema20[-1] and self.bought[s][0] == "LONG":
                        print("CLOSE POSITION: %s" % bar_date)
                        signal = SignalEvent(1, symbol, dt, 'EXIT', 1.0)
                        self.bought[s] = ('OUT', 0)
                        self.events.put(signal)

                    elif bars[-1] + 0.002 <= ema20[-1] and self.bought[s] == "LONG":
                        print("CLOSE POSITION: %s" % bar_date)
                        signal = SignalEvent(1, symbol, dt, 'EXIT', 1.0)
                        self.bought[s] = ('OUT', 0)
                        self.events.put(signal)


                    elif bars[-1] + 0.001 <= ema20[-1] and self.bought[s] == "SHORT":
                        print("CLOSE POSITION: %s" % bar_date)
                        signal = SignalEvent(1, symbol, dt, 'EXIT', 1.0)
                        self.bought[s] = 'OUT'
                        self.events.put(signal)

                    elif bars[-1] - 0.002 >= ema20[-1] and self.bought[s] == "SHORT":
                        print("CLOSE POSITION: %s" % bar_date)
                        signal = SignalEvent(1, symbol, dt, 'EXIT', 1.0)
                        self.bought[s] = 'OUT'
                        self.events.put(signal)


class TrendFollowing(Strategy):
    """
    Trend Following basic strategy w EMA5, EMA12 and RSI21
    """
    def __init__(self, bars, events, window=21):
        """
        Initialises the Moving Average Cross Strategy.
        Parameters:
        bars - The DataHandler object that provides bar information
        events - The Event Queue object.
        short_window - The short moving average lookback.
        long_window - The long moving average lookback.
        """
        self.bars = bars
        self.symbol_list = self.bars.symbol_list
        self.events = events
        self.window = window

        # Set to True if a symbol is in the market
        self.bought = self._calculate_initial_bought()

    def _calculate_initial_bought(self):

        bought = {}
        for s in self.symbol_list:
            bought[s] = ('OUT', 0)
        return bought

    def calculate_signals(self, event):
        # t0 = time.clock()
        # t1 = time.time()
        #     todo min e max de low e hig de certos periodos
        if event.type == 'MARKET':
            for s in self.symbol_list:
                bars = self.bars.get_latest_bars_values(
                    s, "Close", N=100
                )
                low_hist = self.bars.get_latest_bars_values(
                    s, "Low", N=100
                )
                high_hist = self.bars.get_latest_bars_values(
                    s, "High", N=100
                )
                open_hist = self.bars.get_latest_bars_values(
                    s, "Open", N=100
                )

                inputs = {
                    'open': open_hist,
                    'high': high_hist,
                    'low': low_hist,
                    'close': bars
                }
                day = self.bars.get_latest_bars_values(
                    s, "Day", N=21
                )
                time = self.bars.get_latest_bars_values(
                    s, "Time", N=21
                )

                # kf = KalmanFilter(transition_matrices=[1],
                #                   observation_matrices=[1],
                #                   initial_state_mean=0,
                #                   initial_state_covariance=1,
                #                   observation_covariance=1,
                #                   transition_covariance=.01)
                #
                # # Use the observed values of the price to get a rolling mean
                # state_means, _ = kf.filter(bars)


                # bar_date = self.bars.get_latest_bar_datetime(s)
                if bars is not None and bars != [] and len(bars) >= 21:

                    delem = day[-1].split('.')
                    telem = time[-1].split(':')
                    date = datetime(int(float(delem[0])), int(float(delem[1])),int(float(delem[2])), int(float(telem[0])),int(float(telem[1])))
                    weekday = date.weekday()
                    hour = date.hour

                    previousweekday = datetime(int(float(day[-21].split('.')[0])), int(float(day[-21].split('.')[1])),
                                               int(float(day[-21].split('.')[2])), int(float(time[-21].split(':')[0])),
                                               int(float(time[-21].split(':')[1]))).weekday()

                    # sma100 = np.mean(bars[-100:])
                    slope = 0
                    if (weekday == 6):
                        if (previousweekday == 6):

                            ##todo introduzir alternative para negociar as segundas
                            ema5 = EMA(inputs, timeperiod=5)
                            ema12 = EMA(inputs, timeperiod=12)
                            rsi = RSI(inputs, timeperiod=21)
                            # x = range(0, 24)
                            # t = 0
                            # hourly = []
                            # print('hist', len(low_hist))
                            # if len(low_hist) == 1440:
                            #     while t < 24:
                            #         # k = t * 60
                            #         # print('k',k)
                            #
                            #         hourly.append(low_hist[t * 60])
                            #         t = t + 1
                            #     slope = sm.OLS(hourly, x).fit().params[0]
                            #     # slope, intercept, r_value, p_value, std_err = stats.linregress(x, hourly)
                            #     print('SLOPE', slope)
                        else:
                            ema5 = [0]
                            ema12 = [0]
                            rsi = [50]
                            print('FIRST 20 MINUTES of the week')

                    else:
                        ema5 = EMA(inputs, timeperiod=5)
                        ema12 = EMA(inputs, timeperiod=12)
                        rsi = RSI(inputs, timeperiod=21)
                        # x = range(0,24)
                        # t = 0
                        # hourly = []
                        # print('hist', len(low_hist))
                        # if len(low_hist) == 1440:
                        #     while t < 24:
                        #         # k = t * 60
                        #         # print('k',k)
                        #
                        #         hourly.append(low_hist[t * 60])
                        #         t = t + 1
                        #     slope = sm.OLS(hourly, x).fit().params[0]
                        #     # slope, intercept, r_value, p_value, std_err = stats.linregress(x, hourly)
                        #     print('SLOPE', slope)

                        # sma = np.mean(bars[-self.window:])
                        # std = np.std(bars[-self.window:])
                        # bollinger = (bars[len(bars) - 1] - sma) / (2 * std)

                    print(str(weekday), str(hour))
                    weekstop = False
                    if weekday == 4 and hour >= 15:
                        weekstop = True

                    print('weekday', weekday)
                    # ((df.loc[i, 'Close'] - rm[i]) / (2 * df.loc[i, 'STD']))

                    # ax = sma.plot()
                    # std.plot(label='STD', ax=ax)
                    # plt.show()
                    bar_date = self.bars.get_latest_bar_datetime(s)
                    symbol = s
                    dt = bar_date
                    sig_dir = ""
                    stoploss = False
                    print('indicators', ema5[-1], ema12[-1], rsi[-1])
                    if bars[-1] + 0.002 < self.bought[s][1] and bars[-2] - bars[-1] < 0.0004 and self.bought[s][
                        0] == "LONG":
                        stoploss = True

                    elif bars[-1] - 0.002 > self.bought[s][1] and bars[-1] - bars[-2] < 0.0004 and self.bought[s][
                        0] == "SHORT":
                        stoploss = True

                    # print('BOLLINGER', sma)

                    if weekstop == True or stoploss == True:
                        print("CLOSE POSITION: %s" % bar_date)
                        signal = SignalEvent(1, symbol, dt, 'EXIT', 1.0)
                        self.bought[s] = ('OUT', 0)
                        self.events.put(signal)

                    elif weekstop == True or stoploss == True:
                        print("CLOSE POSITION: %s" % bar_date)
                        signal = SignalEvent(1, symbol, dt, 'EXIT', 1.0)
                        self.bought[s] = ('OUT', 0)
                        self.events.put(signal)

                    # print('BOLLINGER', bollinger, 'Self_bought', self.bought[s])
                    elif rsi[-1] > 50 and ema5[-1] > ema12[-1] and self.bought[s][0] == 'OUT':
                        print("LONG: %s" % bar_date)
                        sig_dir = 'LONG'

                        signal = SignalEvent(1, symbol, dt, sig_dir, 1.0)
                        self.events.put(signal)
                        self.bought[s] = ('LONG', bars[-1])

                    elif rsi[-1] < 50 and ema5[-1] < ema12[-1] and self.bought[s][0] == 'OUT':
                        print("SHORT: %s" % bar_date)
                        sig_dir = 'SHORT'

                        signal = SignalEvent(1, symbol, dt, sig_dir, 1.0)
                        self.events.put(signal)
                        self.bought[s] = ('SHORT', bars[-1])


                    elif rsi[-1] >= 75 and self.bought[s][0] == 'LONG':
                        print("CLOSE POSITION: %s" % bar_date)
                        sig_dir = 'EXIT'
                        signal = SignalEvent(1, symbol, dt, sig_dir, 1.0)
                        self.events.put(signal)
                        self.bought[s] = ('OUT', 0)

                    elif rsi[-1] <= 25 and self.bought[s][0] == 'SHORT':
                        print("CLOSE POSITION: %s" % bar_date)
                        sig_dir = 'EXIT'
                        signal = SignalEvent(1, symbol, dt, sig_dir, 1.0)
                        self.events.put(signal)
                        self.bought[s] = ('OUT', 0)




class ForecastingStrategy(Strategy):

    def __init__(self, bars, events):
        self.bars = bars
        self.symbol_list = self.bars.symbol_list
        self.events = events
        self.datetime_now = datetime.utcnow()
        # TODO dates must be set automatically
        self.model_start_date = datetime(2013, 1, 2)
        self.model_end_date = datetime(2013, 12, 20)
        self.model_start_test_date = datetime(2013, 6, 15)
        self.long_market = False
        self.short_market = False
        self.bar_index = 0
        self.model = self.create_symbol_forecast_model()



    def create_symbol_forecast_model(self):

        # Create a lagged series

        snpret = self.bars.create_lagged_series(self.symbol_list[0], self.model_start_date, self.model_end_date, lags=5)

        X = snpret[["Lag1", "Lag2", 'Day']]
        y = snpret[["Direction", 'Day']]

        # Create training and test sets
        # start_test = self.model_start_test_date
        half = (len(snpret) / 4) * 3
        start_test = snpret['Day'][half]
        print(start_test, half,len(snpret))
        X_train = X[X['Day'] < start_test]
        X_test = X[X['Day'] >= start_test]
        y_train = y[y['Day'] < start_test]
        y_test = y[y['Day'] >= start_test]

        model = QDA()
        # print(X_train[["Lag1","Lag2"]])
        # print(y_train['Direction'])
        model.fit(X_train[["Lag1","Lag2"]], y_train['Direction'])
        return model

    def calculate_signals(self, event):
        """
        Calculate the SignalEvents based on market data.
        """

        sym = self.symbol_list[0]
        dt = self.datetime_now
        if event.type == 'MARKET':
            self.bar_index += 1
        if self.bar_index > 5:
                lags = self.bars.get_latest_bars_values(
                    self.symbol_list[0], "Close", N=4
                )
                print('### LAGS ', lags)
                pred_series = pd.Series(
                    {
                'Lag1': lags[1] * 100.0,
                'Lag2': lags[2] * 100.0
                }
                )
                print(pred_series)
                pred = self.model.predict(pred_series)
                if pred > 0 and not self.long_market:
                    self.long_market = True
                    signal = SignalEvent(1, sym, dt, 'LONG', 1.0)
                    self.events.put(signal)
                if pred < 0 and self.long_market:
                    self.long_market = False
                    signal = SignalEvent(1, sym, dt, 'EXIT', 1.0)
                    self.events.put(signal)
                # self.model = self.create_symbol_forecast_model()

class HedgeStrategy(Strategy):
    """
    Uses ordinary least squares (OLS) to perform a rolling linear
    regression to determine the hedge ratio between a pair of equities.
    The z-score of the residuals time series is then calculated in a
    rolling fashion and if it exceeds an interval of thresholds
    (defaulting to [0.5, 3.0]) then a long/short signal pair are generated
    (for the high threshold) or an exit signal pair are generated (for the
    low threshold).
    """

    def __init__(self, bars, events, ols_window=100, zscore_low=0.5, zscore_high=3.0, simulation=True ):
        """
        Initialises the stat arb strategy.
        Parameters:
        bars - The DataHandler object that provides bar information
        events - The Event Queue object.
        """

        self.bars = bars
        self.symbol_list = self.bars.symbol_list
        self.events = events
        self.ols_window = ols_window
        self.zscore_low = zscore_low
        self.zscore_high = zscore_high
        self.pair = ('eur/USD','eur/AUD' )
        self.datetime = datetime.utcnow()
        self.long_market = False
        self.short_market = False
        self.bought = self._calculate_initial_bought()
        self.simulation = simulation
        self.daycounter = 0

    def _calculate_initial_bought(self):

        bought = {}
        for s in self.symbol_list:
            bought[s] = ('OUT',0)
        return bought

    def calculate_xy_signals(self, zscore_last, prices):
        """
        Calculates the actual x, y signal pairings
        to be sent to the signal generator.   

        Parameters
        zscore_last - The current zscore
        
        """
        y_signal = None
        x_signal = None
        p0 = self.pair[0]
        p1 = self.pair[1]
        day = self.bars.get_latest_bars_values(
            p0, "Day", N=27
        )

        time = self.bars.get_latest_bars_values(
            p0, "Time", N=27
        )
        # dt = self.datetime
        dt0 = self.bars.get_latest_bar_datetime(p0)
        dt1 = self.bars.get_latest_bar_datetime(p1)
        hr = abs(self.hedge_ratio)

        delem = day[-1].split('.')
        telem = time[-1].split(':')
        date = datetime(int(float(delem[0])), int(float(delem[1])), int(float(delem[2])), int(float(telem[0])),
                        int(float(telem[1])))
        weekday = date.weekday()
        hour = date.hour
        weekstop = False
        if float(weekday) == 4 and hour > 16:
            weekstop = True
        dt = dt0
        # print(prices[1][-1])
        stoploss = False
        # if prices[0][-1] + 0.002 < self.bought[p0][1] and prices[0][-2] - prices[0][-1] < 0.0004 and self.bought[p0][0] == "LONG":
        #     stoploss = True
        #
        # elif prices[0][-1] - 0.002 > self.bought[p0][1] and prices[0][-1] - prices[0][-2] < 0.0004 and self.bought[p0][0] == "SHORT":
        #     stoploss = True
        #
        # elif prices[1][-1] + 0.002 < self.bought[p1][1] and prices[1][-2] - prices[1][-1] < 0.0004 and self.bought[p1][0] == "LONG":
        #     stoploss = True
        #
        # elif prices[1][-1] - 0.002 > self.bought[p1][1] and prices[1][-1] - prices[1][-2] < 0.0004 and self.bought[p1][0] == "SHORT":
        #     stoploss = True

        if (weekstop == True or stoploss == True) and self.bought[p0][0] == "LONG":
            print("CLOSE POSITION: %s" % dt)
            y_signal = SignalEvent(1, p0, dt0, 'EXIT', 1.0)
            x_signal = SignalEvent(1, p1, dt1, 'EXIT', 1.0)
            self.bought[p0] = ('OUT', 0)
            self.bought[p1] = ('OUT', 0)

        elif (weekstop == True or stoploss == True) and self.bought[p0][0] == "SHORT":
            print("CLOSE POSITION: %s" % dt)
            y_signal = SignalEvent(1, p0, dt0, 'EXIT', 1.0)
            x_signal = SignalEvent(1, p1, dt1, 'EXIT', 1.0)
            self.bought[p0] = ('OUT', 0)
            self.bought[p1] = ('OUT', 0)

        # If we're long the market and below the
        # negative of the high zscore threshold
        if zscore_last <= -self.zscore_high and not self.long_market:
            self.bought[p1] = ('SHORT', prices[1][-1])
            self.bought[p0] = ('LONG', prices[0][-1])
            self.long_market = True
            y_signal = SignalEvent(1, p0, dt0, 'LONG', 1.0)
            x_signal = SignalEvent(1, p1, dt1, 'SHORT', hr)
            # If we're long the market and between the
            # absolute value of the low zscore threshold
        elif abs(zscore_last) <= self.zscore_low and self.long_market:
            self.bought[p0] = ('OUT', 0)
            self.bought[p1] = ('OUT', 0)
            self.long_market = False
            y_signal = SignalEvent(1, p0, dt0, 'EXIT', 1.0)
            x_signal = SignalEvent(1, p1, dt1, 'EXIT', 1.0)
            # If we're short the market and above
            # the high zscore threshold
        elif zscore_last >= self.zscore_high and not self.short_market:
            self.bought[p0] = ('SHORT', prices[0][-1])
            self.bought[p1] = ('LONG', prices[1][-1])
            self.short_market = True
            y_signal = SignalEvent(1, p0, dt0, 'SHORT', 1.0)
            x_signal = SignalEvent(1, p1, dt1, 'LONG', hr)
            # If we're short the market and between the
            # absolute value of the low zscore threshold
        elif abs(zscore_last) <= self.zscore_low and self.short_market:
            self.bought[p0] = ('OUT', 0)
            self.bought[p1] = ('OUT', 0)
            self.short_market = False
            y_signal = SignalEvent(1, p0, dt0, 'EXIT', 1.0)
            x_signal = SignalEvent(1, p1, dt1, 'EXIT', 1.0)

        return y_signal, x_signal

    def calculate_signals(self, event):
        """
        Generates a new set of signals based on the mean reversion
        strategy.
        Calculates the hedge ratio between the pair of tickers.
        We use OLS for this, althought we should ideall use CADF.
        """
        # Obtain the latest window of values for each
        # component of the pair of tickers
        if event.type == 'MARKET':
            y = self.bars.get_latest_bars_values(
                self.pair[0], "Close", N=self.ols_window
            )
            x = self.bars.get_latest_bars_values(
                self.pair[1], "Close", N=self.ols_window
            )
            if y is not None and x is not None:
                # Check that all window periods are available
                if len(y) >= self.ols_window and len(x) >= self.ols_window:
                    # Calculate the current hedge ratio using OLS
                    self.hedge_ratio = sm.OLS(y, x).fit().params[0]
                    # Calculate the current z-score of the residuals
                    spread = y - self.hedge_ratio * x
                    # hurstv = hurst(spread)
                    # print('Hurst', hurstv)
                    zscore_last = ((spread - spread.mean()) / spread.std())[-1]
                    # Calculate signals and add to events queue
                    prices = [y,x]
                    y_signal, x_signal = self.calculate_xy_signals(zscore_last, prices)
                    if y_signal is not None and x_signal is not None:
                        self.events.put(y_signal)
                        self.events.put(x_signal)