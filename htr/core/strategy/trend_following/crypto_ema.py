
import talib

from htr.core.events import *
from htr.core.strategy import Strategy


class CryptoEma(Strategy):
    """
       Carries out a basic Moving Average Crossover strategy with a
       short/long simple weighted moving average. Default short/long
       windows are 20/100 periods respectively.
    """

    def __init__(self, context, events, data_handler):
        """
        Initialises the Moving Averages Strategy.

        Args:
            context (Context): Runtime context (e.g. Strategies, initial_capital, timeframe, etc).
            events (Queue): Event queue object where signals are shared.
            data_handler (DataHandler) - The DataHandler object that provides bar information.
            short_window - The short moving average lookback.
            long_window - The long moving average lookback.
            trend_window - The window used to detect trends with the moving average.
        """
        self.data_handler = data_handler
        self.symbol_list = self.data_handler.symbol_list
        self.events = events
        self.short_window = int(context.fast_ema)
        self.long_window = int(context.slow_ema)
        self.bought = self._calculate_initial_bought()
        self.trend_flag = False

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
            symbol, "Close", N=601
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
            if data is not None and len(data) > 100:


                ema_fast = talib.EMA(data, timeperiod=self.short_window)
                ema_slow = talib.EMA(data, timeperiod=self.long_window)

                if ema_slow[-1] > ema_fast[-1] and self.bought[symbol][0] == "OUT":
                    print("LONG: %s" % bar_date)
                    self.trend_flag = True
                    # Create BUY signal.
                    signal = SignalEvent(1, symbol, bar_date, 'LONG', 1.0)
                    # Update bought status in strategy position cache.
                    self.bought[symbol] = ('LONG', data[-1])
                    # Share signal in the events queue.
                    self.events.put(signal)
                # If short moving average is lower than the long moving average lets EXIT position.
                elif ema_slow[-1] < ema_fast[-1] and self.bought[symbol][0] == "LONG":
                    print("CLOSE POSITION: %s" % bar_date)
                    # Create EXIT signal.
                    signal = SignalEvent(1, symbol, bar_date, 'EXIT', 1.0)
                    # Update bought status in strategy position cache.
                    self.bought[symbol] = ('OUT', data[-1])
                    # Share signal in the events queue.
                    self.events.put(signal)
