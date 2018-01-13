import numpy as np
from pykalman import KalmanFilter
import pandas as pd
import talib

from htr.core.events import *
from htr.core.strategy import Strategy


class CryptoReturnsReversal(Strategy):
	"""
	  .
	   """

	def __init__(self, context, events, data_handler, short_window=21, long_window=100):
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
			symbol, "Close", N=self.long_window
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
			if data is not None and len(data) == 100:

				data = pd.Series(data).pct_change()[1:].values
				"""
				kf = KalmanFilter(transition_matrices=[1],
								  observation_matrices=[1],
								  initial_state_mean=0,
								  initial_state_covariance=1,
								  observation_covariance=1,
								  transition_covariance=.01)

				# Use the observed values of the price to get a rolling mean
				state_means, _ = kf.filter(data)
				"""
				ema12 = talib.EMA(data, timeperiod=12)
				std = np.std(data) / 3
				upperband = std + ema12
				lowerband = ema12 - std
				"""
				print('BBWidth : ', upperband[-1] - lowerband[-1])
				if upperband[-1] is not None:
					bbwidth = (upperband[-1] - lowerband[-1])*100
					self.bb_width.append(bbwidth)
					print('BBWidth avg: ', sum(self.bb_width) / len(self.bb_width))
					print('BBWidth min: ', min(self.bb_width))
					print('BBWidth max: ', max(self.bb_width))
				"""

				# If short moving average is higher than the long moving average lets BUY.
				if data[-1] <= lowerband[-1] and self.bought[symbol][0] == "OUT":
					print("LONG: %s" % bar_date)
					# Create BUY signal.
					signal = SignalEvent(1, symbol, bar_date, 'LONG', 1.0)
					# Update bought status in strategy position cache.
					self.bought[symbol] = ('LONG', data[-1])
					# Share signal in the events queue.
					self.events.put(signal)

				# If short moving average is lower than the long moving average lets EXIT position.
				elif data[-1] >= upperband[-1] and self.bought[symbol][0] == "LONG":
					print("CLOSE POSITION: %s" % bar_date)
					# Create EXIT signal.
					signal = SignalEvent(1, symbol, bar_date, 'EXIT', 1.0)
					# Update bought status in strategy position cache.
					self.bought[symbol] = ('OUT', data[-1])
					# Share signal in the events queue.
					self.events.put(signal)
