
import statsmodels.api as sm
from datetime import datetime as dt
try:
    import Queue as queue
except ImportError:
    import queue

import numpy as np
import pandas as pd
from htr.core.events import GroupSignalEvent, SignalEvent
from htr.core.strategy import Strategy

class PairsTrading(Strategy):
    """
    Uses ordinary least squares (OLS) to perform a rolling linear
    regression to determine the hedge ratio between a pair of equities.
    The z-score of the residuals time series is then calculated in a
    rolling fashion and if it exceeds an interval of thresholds
    (defaulting to [0.5, 3.0]) then a long/short signal pair are generated
    (for the high threshold) or an exit signal pair are generated (for the
    low threshold).
    """

    def __init__(self, context, events, data_handler, ols_window=100, zscore_low=0.5, zscore_high=3.0):
        """
        Initialises the stat arb strategy.
        Parameters:
        data_handler - The DataHandler object that provides bar information
        events - The Event Queue object.
        """

        self.data_handler = data_handler
        self.symbol_list = self.data_handler.symbol_list
        self.events = events

        self.ols_window = int(context.ols_window)
        self.zscore_low = float(context.zscore_low)
        self.zscore_high = float(context.zscore_high)
        self.pair = (context.data_sources[0]['symbol'], context.data_sources[1]['symbol'])
        self.datetime = dt.utcnow()
        self.long_market = False
        self.short_market = False

    def calculate_xy_signals(self, zscore_last):
        """
        Calculates the actual x, y signal pairings
        to be sent to the signal generator.

        Parameters
        zscore_last - The current zscore

        """

        p0 = self.pair[0]
        p1 = self.pair[1]
        # dt = self.datetime
        dt0 = self.data_handler.get_latest_bar_datetime(p0)
        dt1 = self.data_handler.get_latest_bar_datetime(p1)
        hr = abs(self.hedge_ratio)
        # If we're long the market and below the
        # negative of the high zscore threshold
        if zscore_last <= -self.zscore_high and not self.long_market:
            self.long_market = True
            ## todo put group signal
            y_signal = SignalEvent(1, p0, dt0, 'LONG', 1.0)
            x_signal = SignalEvent(1, p1, dt1, 'SHORT', hr)

            signal = GroupSignalEvent(1, [y_signal, x_signal], dt.utcnow(), 1)
            return signal
            # If we're long the market and between the
            # absolute value of the low zscore threshold
        elif abs(zscore_last) <= self.zscore_low and self.long_market:
            self.long_market = False
            y_signal = SignalEvent(1, p0, dt0, 'EXIT', 1.0)
            x_signal = SignalEvent(1, p1, dt1, 'EXIT', 1.0)

            signal = GroupSignalEvent(1, [y_signal, x_signal], dt.utcnow(), 1)
            return signal
            # If we're short the market and above
            # the high zscore threshold
        elif zscore_last >= self.zscore_high and not self.short_market:
            self.short_market = True
            y_signal = SignalEvent(1, p0, dt0, 'SHORT', 1.0)
            x_signal = SignalEvent(1, p1, dt1, 'LONG', hr)

            signal = GroupSignalEvent(1, [y_signal, x_signal], dt.utcnow(), 1)
            return signal
            # If we're short the market and between the
            # absolute value of the low zscore threshold
        elif abs(zscore_last) <= self.zscore_low and self.short_market:
            self.short_market = False
            y_signal = SignalEvent(1, p0, dt0, 'EXIT', 1.0)
            x_signal = SignalEvent(1, p1, dt1, 'EXIT', 1.0)

            signal = GroupSignalEvent(1, [y_signal, x_signal], dt.utcnow(), 1)
            return signal

    def calculate_signals(self):
        """
        Generates a new set of signals based on the mean reversion
        strategy.
        Calculates the hedge ratio between the pair of tickers.
        We use OLS for this, althought we should ideall use CADF.
        """
        # Obtain the latest window of values for each
        # component of the pair of tickers

        y = self.data_handler.get_latest_bars_values(
            self.pair[0], "Close", N=self.ols_window
        )
        x = self.data_handler.get_latest_bars_values(
            self.pair[1], "Close", N=self.ols_window
        )

        if y is not None and x is not None:
            # Check that all window periods are available
            if len(y) >= self.ols_window and len(x) >= self.ols_window:
                # Calculate the current hedge ratio using OLS
                try:
                    self.hedge_ratio = sm.OLS(y, x).fit().params[0]
                except Exception as e:
                    print('wtf ols? ', e.__str__())
                    return
                # Calculate the current z-score of the residuals
                spread = y - self.hedge_ratio * x
                zscore_last = ((spread - spread.mean()) / spread.std())[-1]
                # Calculate signals and add to events queue
                signal = self.calculate_xy_signals(zscore_last)
                if signal is not None:
                    self.events.put(signal)
