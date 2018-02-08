
import talib
import pandas as pd

from htr.core.events import *
from htr.core.strategy import Strategy



class CryptoBull(Strategy):
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
		self.symbol_list = self.data_handler.symbol_list
		self.events = events
		self.short_window = short_window
		self.long_window = long_window
		self.bought = self._calculate_initial_bought()
		self.bb_width = []
		self.pos_count = {}
		for s in self.symbol_list:
			self.pos_count[s] = 0

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
			if data is not None and len(data) >= 100:
				# returns = pd.Series(data).pct_change()
				# sum_returns = sum(returns.values[-3:])
				# ret = (data[-1] - self.bought[symbol][1]) / self.bought[symbol][1]
				# print("Data: ", data)
				slope = talib.LINEARREG_SLOPE(data, timeperiod=14)
				# if self.bought[symbol][1] != 0:
				# 	ret = (slope[-1] - self.bought[symbol][1]) / self.bought[symbol][1]
				upperband, middleband, lowerband = talib.BBANDS(slope, timeperiod=20, nbdevup=1, nbdevdn=1, matype=0)
				print('Slope: ', slope[-1])
				print('Price: ', data[-1])
				# print('Ret : ', ret)
				## 87
				# if slope[-1] >= lowerband[-1] and slope[-2] < lowerband[-1] and self.pos_count[symbol] == 0:

				## 48
				if slope[-1] <= lowerband[-1] and slope[-2] < slope[-1] and slope[-1] > -55  and self.pos_count[symbol] == 0:
					self.pos_count[symbol] += 1
					print("LONG: %s" % bar_date)
					self.bought[symbol] = ('', slope[-1])
					# Create BUY signal.
					signal = SignalEvent(1, symbol, bar_date, 'LONG', 1.0)
					# Share signal in the events queue.
					self.events.put(signal)
				## todo lucrar no upper band

				elif slope[-1] >= upperband[-1] and self.pos_count[symbol] > 0:
					self.pos_count[symbol] -= 1
					print("CLOSE POSITION: %s" % bar_date)
					self.bought[symbol] = ('', 0)
					# Create EXIT signal.
					signal = SignalEvent(1, symbol, bar_date, 'EXIT', 1.0)
					# Share signal in the events queue.
					self.events.put(signal)
				"""
				elif self.pos_count[symbol] > 0 and ret >= 2 and slope[-1] < 0:
					self.pos_count[symbol] -= 1
					print("CLOSE POSITION: %s" % bar_date)
					self.bought[symbol] = ('', 0)
					# Create EXIT signal.
					signal = SignalEvent(1, symbol, bar_date, 'EXIT', 1.0)
					# Share signal in the events queue.
					self.events.put(signal)
				"""

