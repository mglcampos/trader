
import talib
import statsmodels.tsa.stattools as ts
import numpy as np
import matplotlib.pyplot as plt
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

		self.hurst_cache = []
		self.adf_cache = []

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
		# linear fit to double-log graph (gives power)
		m = np.polyfit(np.log10(lagvec), np.log10(tau), 1)
		# calculate hurst
		hurst = m[0] * 2
		return hurst

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
				ema12 = talib.EMA(data, timeperiod=12)

				slope_pos = ''
				if slope[-1] <= lowerband[-1]:
					slope_pos = 'BELOW LOWERBAND'
				elif slope[-1] >= upperband[-1]:
					slope_pos = 'ABOVE UPPERBAND'
				elif slope[-1] >= middleband[-1]:
					slope_pos = 'ABOVE MIDDLEBAND'

				print('Slope: ', slope[-1], slope_pos)
				print('Price: ', data[-1])

				# cadf = ts.adfuller(data)
				# print('ADF: ', cadf)
				# print('HURST: ', self.hurst(data))
				self.hurst_cache.append(self.hurst(data))
				self.adf_cache.append(ts.adfuller(data)[0])

				if len(self.hurst_cache) > 800:
					fig = plt.figure(2)
					fig.suptitle('Stationarity', fontsize=16)
					ax = plt.subplot(211)
					ax.title.set_text('ADF')
					pd.Series(self.adf_cache).plot(legend=None)
					ax = plt.subplot(212)
					ax.title.set_text('Hurst')
					pd.Series(self.hurst_cache).plot(legend=None)


				if slope[-1] >= lowerband[-1] and slope[-2] < lowerband[-1] and data[-1] <= ema12[-1] and self.pos_count[symbol] == 0:
					strength = 1.0
					# if data[-1] <= ema12[-1]:
					# 	strength = 1.0
					# else:
					# 	strength = 0.5

					self.pos_count[symbol] += 1
					print("LONG: %s" % bar_date)
					self.bought[symbol] = ('', slope[-1])
					# Create BUY signal.
					signal = SignalEvent(1, symbol, bar_date, 'LONG', strength)
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
