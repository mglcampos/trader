
import numpy as np

from htr.core.events import *
from htr.core.strategy import Strategy
from talib.abstract import *


class ADXCross(Strategy):
	"""
	Carries out a basic Moving Average Crossover strategy with a
	short/long simple weighted moving average. Default short/long
	windows are 100/400 periods respectively.
	"""

	def __init__(self, bars, events, short_window=9, long_window=21, simulation=True):
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
			bought[s] = ('OUT', 0)

		return bought

	def _load_data(self, symbol):
		"""Imports data using the DataHandler.
		Args:
			symbol:

		Returns:

		"""
		close = self.bars.get_latest_bars_values(
			symbol, "Close", N=100
		)
		high = self.bars.get_latest_bars_values(
			symbol, "High", N=100
		)
		low = self.bars.get_latest_bars_values(
			symbol, "Low", N=100
		)
		data = {
			'high': np.array(high),
			'low': np.array(low),
			'close': np.array(close)
		}

		return data

	def _check_stop(self):
		"""Check for stop conditions, e.g. Stop losses, take profit, etc

		Generates EXIT signals.

		Returns:
			bool: True if EXIT signal was issued, False if it was not.
		"""

		pass

	def calculate_signals(self, event):

		if event.type == 'MARKET':
			for symbol in self.symbol_list:

				data = self._load_data(symbol)
				if self._check_stop():
					return

				adx = ADX(data, timeperiod=14)

				print('ADX', adx[-1])
				if data['close'] is not None:
					short_sma = np.mean(data['close'][-self.short_window:])
					long_sma = np.mean(data['close'][-self.long_window:-2])  ##shifted by a factor of 2
					dt = self.bars.get_latest_bar_datetime(symbol)
					print("self_bought", self.bought[symbol], symbol)
					if short_sma > long_sma and adx[-1] >= 22 and self.bought[symbol][0] == "OUT":
						print("LONG: %s" % dt)
						signal = SignalEvent(1, symbol, dt, 'LONG', 1.0)
						self.bought[symbol] = ('LONG', data['close'][-1])
						self.events.put(signal)


					elif short_sma < long_sma and self.bought[symbol][0] == "LONG":
						print("CLOSE POSITION: %s" % dt)
						signal = SignalEvent(1, symbol, dt, 'EXIT', 1.0)
						self.bought[symbol] = ('OUT', data['close'][-1])
						self.events.put(signal)