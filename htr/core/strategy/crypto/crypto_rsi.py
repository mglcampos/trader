import numpy as np
from talib.abstract import *

from htr.core.strategy import Strategy
from htr.core.events import SignalEvent


class CryptoRsi(Strategy):
	"""
	Uses ADX and RSI value to calculate signals and ATR to calculate stoplosses
	"""

	def __init__(self, context, events, data_handler, short_window=9, long_window=21):
		"""
		Initialises the Moving Average Cross Strategy.
		Parameters:
		bars - The DataHandler object that provides bar information
		events - The Event Queue object.
		short_window - The short moving average lookback.
		long_window - The long moving average lookback.
		"""
		self.data_handler = data_handler
		self.symbol_list = self.data_handler.symbol_list
		self.events = events
		self.short_window = short_window
		self.long_window = long_window
		self.bought = self._calculate_initial_bought()

	def _calculate_initial_bought(self):

		bought = {}
		for s in self.symbol_list:
			bought[s] = ('OUT', 0)
		return bought

	def calculate_signals(self):

		for s in self.symbol_list:

			close = self.data_handler.get_latest_bars_values(
				s, "Close", N=100
			)
			high = self.data_handler.get_latest_bars_values(
				s, "High", N=100
			)
			low = self.data_handler.get_latest_bars_values(
				s, "Low", N=100
			)
			inputs = {
				'high': np.array(high),
				'low': np.array(low),
				'close': np.array(close)
			}
			# adx = ADX(inputs, timeperiod=14)
			rsi5 = RSI(inputs, timeperiod=5)
			# rsi21 = RSI(inputs, timeperiod=21)
			# atr = ATR(inputs)

			# print('ADX', adx[-1])
			# print('rsi21', rsi21[-1])
			# print('rsi5', rsi5[-1])
			if close is not None:
				# short_sma = np.mean(close[-self.short_window:])
				# long_sma = np.mean(close[-self.long_window:-2]) ##shifted by a factor of 2
				end = False
				if rsi5[-1] >= 70:
					end = True
				stoploss = False
				## todo change pip loss to crypto scenario
				# pip_loss = atr[-1] * 0.0001 * 2  ##todo make sure is always positive
				# if close[-1] + pip_loss < self.bought[s][1]  and close[-2] - close[-1] < 0.0004 and self.bought[s][
				# 	0] == "LONG":
				# 	stoploss = True
				#
				# elif close[-1] - pip_loss > self.bought[s][1] and close[-1] - close[-2] < 0.0004 and self.bought[s][
				# 	0] == "SHORT":
				# 	stoploss = True
				symbol = s
				dt = self.data_handler.get_latest_bar_datetime(s)
				print("self_bought", self.bought[s], s)
				if rsi5[-1] <= 30 and self.bought[s][0] == "OUT":
				# if rsi21[-1] > 30 and rsi5[-1] > 30 and adx[-1] >= 20 and self.bought[s][0] == "OUT":
					print("LONG: %s" % dt)
					signal = SignalEvent(1, symbol, dt, 'LONG', 1.0)
					self.bought[s] = ('LONG', close[-1])
					self.events.put(signal)


				elif (end == True or stoploss == True) and self.bought[s][0] == "LONG":
					print("CLOSE POSITION: %s" % dt)
					signal = SignalEvent(1, symbol, dt, 'EXIT', 1.0)
					self.bought[s] = ('OUT', 0)
					self.events.put(signal)
