
import talib

import numpy as np
import pandas as pd
import sklearn.mixture as mix
from sklearn import preprocessing
from pykalman import KalmanFilter
from htr.core.events import *
from htr.core.strategy import Strategy
from htr.helpers.ml import Classifier
from htr.helpers import Welford



class GaussianMix(Strategy):
    """
      .
       """

    def __init__(self, context, events, data_handler, slow_period=21, fast_period=7):
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

        self.slow_period = slow_period
        self.fast_period = fast_period
        self.minimum_data_size = 100
        self.bought = self._calculate_initial_bought()
        self.welford = Welford(3)
        self.pos_count = {}
        self.signal = 0.0
        for s in self.symbol_list:
            self.pos_count[s] = 0


        self.pred = 0
        self.fprice = 0.0
        self.ticker = self.symbol_list[0]


        ## Add big sample
        X = pd.read_csv('C:\\Users\\utilizador\\Documents\\quant_research\\data\\basic_{}_sample.csv'.format(self.ticker.lower()))[-1500:]
        self.last_bar_date = pd.to_datetime(X.Date).values[-1]
        X = X.drop(['Date'], axis=1)
        X = X[['Returns', 'FReturns', 'Hurst', 'Corr', 'High', 'Low', 'Open', 'Close', 'FClose', 'RSI', 'Stoch_Osc']]
        # print(X.head())
        self.ss = preprocessing.StandardScaler()
        self.X = X
        self.unsup = mix.GaussianMixture(n_components=4, covariance_type="spherical", n_init=100, random_state=42)
        self.unsup.fit(np.reshape(self.ss.fit_transform(self.X), (-1, self.X.shape[1])))

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
        if weekstop:
            print('Weekstop')
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

    def filter_prices(self, data):
        kf = KalmanFilter(transition_matrices=[1],
                          observation_matrices=[1],
                          initial_state_mean=data[0],
                          initial_state_covariance=1,
                          observation_covariance=1,
                          transition_covariance=.01)

        state_means, _ = kf.filter(data)
        state_means = state_means.flatten()
        return state_means

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
            if  pd.to_datetime([bar_date]).values[0] <= self.last_bar_date:
                print('Training sample start  date: {}, backtest start date: {}'.format(self.last_bar_date, bar_date))
                return
            # Perform technical analysis.
            if data is not None and len(data) >= self.minimum_data_size:
                vol = self.welford.update_and_return(data[-3:]) * 100
                # Checks for stop conditions
                if self._check_week_stop(bar_date) or self._check_stop(data):
                    self._fire_sale(bar_date, data)
                    print("Fire sale")
                    return

                if self._check_spread_stop(bar_date):
                    print('Spread stop')
                    return



                high = self.data_handler.get_latest_bars_values(
                    symbol, "High", N=1
                )
                low = self.data_handler.get_latest_bars_values(
                    symbol, "Low", N=1
                )
                open = self.data_handler.get_latest_bars_values(
                    symbol, "Open", N=1
                )
                ufhurst = pd.DataFrame(data[-21:], columns=['Hurst']).rolling(self.slow_period).apply(self.hurst)
                corr = pd.DataFrame(data[-21:], columns=['Corr']).rolling(self.slow_period).apply(
                    lambda x: x.autocorr(), raw=False)
                ret = pd.DataFrame(data[-2:]).pct_change().values[-1]
                state_means = self.filter_prices(data[-100:])
                fret = pd.DataFrame(state_means[-2:]).pct_change().values[-1]
                rolling_min = pd.DataFrame(state_means).rolling(self.fast_period).min()
                rolling_max = pd.DataFrame(state_means).rolling(self.fast_period).max()
                sample = pd.concat(
                    [pd.DataFrame(state_means), pd.DataFrame(data), rolling_min, rolling_max, ufhurst, corr], axis=1)

                sample.columns = ['FPrice', 'Price', 'Min', 'Max','Hurst','Corr']
                stoch_osc = (sample.FPrice - sample.Min) / (sample.Max - sample.Min)
                rsi = pd.DataFrame(talib.RSI(state_means, timeperiod=self.fast_period), columns=['RSI']).values

                row = [ret[-1], fret[-1], ufhurst.values[-1][0], corr.values[-1][0], high[0], low[0], open[0], data[-1], state_means[-1], rsi[-1][0], stoch_osc.values[-1]]
                print("Row: {}".format(row))
                print(len(row), len(self.X.columns))
                print("Iter: {}, data: {}".format(self.pred, data[-1]))
                self.X = self.X.append(pd.DataFrame([row], columns=self.X.columns), ignore_index=True)
                self.X = self.X.iloc[1:]
                self.X.index = self.X.index.values + self.pred

                try:
                    if self.pred == 0 or self.pred % 300 == 0:
                        self.unsup.fit(np.reshape(self.ss.fit_transform(self.X), (-1, self.X.shape[1])))
                    reshaped = np.reshape(self.ss.fit_transform(self.X[-1:]), (-1, self.X.shape[1]))
                    regime = self.unsup.predict(reshaped)[0]
                except:
                    ## todo improve this
                    self.X.iloc[-1] = self.X.iloc[-2]
                    print("NaNs in training data prob.")
                    return

                bbh = state_means[-1] + 2*vol / 100
                bbl = state_means[-1] - 2*vol / 100
                covs=sorted(zip([0,1,2,3],self.unsup.covariances_), key=lambda x: x[1])
                print("Regimes: {}, Covariances: {}".format(regime, covs[:2]))
                low_cov_regimes = [list(t) for t in zip(*covs)][0][:2]
                reverting = False
                #predict, check low cov regimes, create rule
                if regime in low_cov_regimes:
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

                elif state_means[-1] <= data[-1] + vol / 100 and self.bought[symbol][0] == 'LONG':
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

                elif state_means[-1] >= data[-1] + vol / 100 and self.bought[symbol][0] == 'SHORT':
                    self.signal = 0.0
                    print("CLOSE POSITION: %s" % bar_date)
                    signal = SignalEvent(1, symbol, bar_date, 'EXIT', 1.0)
                    self.bought[symbol] = ('OUT', data[-1])
                    self.events.put(signal)

                self.pred += 1
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


