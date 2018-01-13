import statsmodels.api as sm
from datetime import timedelta, datetime
from pykalman import KalmanFilter

try:
	import Queue as queue
except ImportError:
	import queue

import numpy as np
import pandas as pd
from htr.core.events import *
from htr.core.strategy import Strategy
from talib.abstract import *


class ADXRSIVolatility(Strategy):
	"""
	Uses ADX and RSI value to calculate signals and ATR to calculate stoplosses
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

	def calculate_signals(self, event):

		if event.type == 'MARKET':
			for s in self.symbol_list:

				close = self.bars.get_latest_bars_values(
					s, "Close", N=100
				)
				high = self.bars.get_latest_bars_values(
					s, "High", N=100
				)
				low = self.bars.get_latest_bars_values(
					s, "Low", N=100
				)
				inputs = {
					'high': np.array(high),
					'low': np.array(low),
					'close': np.array(close)
				}
				adx = ADX(inputs, timeperiod=14)
				rsi5 = RSI(inputs, timeperiod=5)
				rsi21 = RSI(inputs, timeperiod=21)
				atr = ATR(inputs)

				print('ADX', adx[-1])
				print('rsi5', rsi21[-1])
				print('rsi21', rsi5[-1])
				if close is not None:
					# short_sma = np.mean(close[-self.short_window:])
					# long_sma = np.mean(close[-self.long_window:-2]) ##shifted by a factor of 2
					end = False
					if rsi21[-1] > 70:
						end = True
					stoploss = False
					pip_loss = atr[-1] * 0.0001 * 2  ##todo make sure is always positive
					if close[-1] + pip_loss < self.bought[s][1] and close[-2] - close[-1] < 0.0004 and self.bought[s][
						0] == "LONG":
						stoploss = True

					elif close[-1] - pip_loss > self.bought[s][1] and close[-1] - close[-2] < 0.0004 and self.bought[s][
						0] == "SHORT":
						stoploss = True
					symbol = s
					dt = self.bars.get_latest_bar_datetime(s)
					print("self_bought", self.bought[s], s)
					# if short_sma > long_sma and adx[-1] >= 20 and self.bought[s][0] == "OUT":
					if rsi21[-1] > 30 and rsi5[-1] > 30 and adx[-1] >= 20 and self.bought[s][0] == "OUT":
						print("LONG: %s" % dt)
						signal = SignalEvent(1, symbol, dt, 'LONG', 1.0)
						self.bought[s] = ('LONG', close[-1])
						self.events.put(signal)


					# elif short_sma < long_sma and self.bought[s][0] == "LONG":
					elif (end == True or stoploss == True) and self.bought[s][0] == "LONG":
						print("CLOSE POSITION: %s" % dt)
						signal = SignalEvent(1, symbol, dt, 'EXIT', 1.0)
						self.bought[s] = ('OUT', 0)
						self.events.put(signal)


class KalmanBandsStrategy(Strategy):
	"""
	Carries out a basic Bollinger Bands strategy
	"""

	def __init__(self, bars, events, window=100):
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
			bought[s] = 'OUT'
		return bought

	def calculate_signals(self, event):

		if event.type == 'MARKET':
			for s in self.symbol_list:
				bars = self.bars.get_latest_bars_values(
					s, "Close", N=self.window
				)
				bar_date = self.bars.get_latest_bar_datetime(s)
				if bars is not None and bars != []:
					# sma = np.mean(bars[-self.window:])
					# std = np.std(bars[-self.window:])

					# ((df.loc[i, 'Close'] - rm[i]) / (2 * df.loc[i, 'STD']))

					# ax = sma.plot()
					# std.plot(label='STD', ax=ax)
					# plt.show()

					kf = KalmanFilter(transition_matrices=[1],
					                  observation_matrices=[1],
					                  initial_state_mean=0,
					                  initial_state_covariance=1,
					                  observation_covariance=1,
					                  transition_covariance=.01)

					# Use the observed values of the price to get a rolling mean
					state_means, _ = kf.filter(bars)
					std = np.std(bars[-20:])

					bar_date = self.bars.get_latest_bar_datetime(s)
					symbol = s
					dt = bar_date
					sig_dir = ""
					bollinger = (bars[len(bars) - 1] - state_means[-1][0]) / (2 * std)

					print('BOLLINGER', bollinger)
					if bollinger < -1.0 and self.bought[s] == 'OUT':
						print("LONG: %s" % bar_date)
						sig_dir = 'LONG'

						signal = SignalEvent(1, symbol, dt, sig_dir, 1.0)
						self.events.put(signal)
						self.bought[s] = 'LONG'

					elif bollinger > 1.0 and self.bought[s] == 'OUT':
						print("SHORT: %s" % bar_date)
						sig_dir = 'SHORT'

						signal = SignalEvent(1, symbol, dt, sig_dir, 1.0)
						self.events.put(signal)
						self.bought[s] = 'SHORT'


					elif bollinger > -0.3 and self.bought[s] == 'LONG':
						print("CLOSE POSITION: %s" % bar_date)
						sig_dir = 'EXIT'
						signal = SignalEvent(1, symbol, dt, sig_dir, 1.0)
						self.events.put(signal)
						self.bought[s] = 'OUT'

					elif bollinger < 0.3 and self.bought[s] == 'SHORT':
						print("CLOSE POSITION: %s" % bar_date)
						sig_dir = 'EXIT'
						signal = SignalEvent(1, symbol, dt, sig_dir, 1.0)
						self.events.put(signal)
						self.bought[s] = 'OUT'


class BollingerBandsStrategy(Strategy):
	"""
	Carries out a basic Bollinger Bands strategy
	"""

	def __init__(self, bars, events, window=20):
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

		if event.type == 'MARKET':
			for s in self.symbol_list:
				bars = self.bars.get_latest_bars_values(
					s, "Close", N=100
				)

				day = self.bars.get_latest_bars_values(
					s, "Day", N=21
				)
				time = self.bars.get_latest_bars_values(
					s, "Time", N=21
				)

				if bars is not None and bars != [] and len(bars) >= 100:

					# todo tirar os splits
					delem = day[-1].split('.')
					telem = time[-1].split(':')
					date = datetime(int(float(delem[0])), int(float(delem[1])), int(float(delem[2])),
					                int(float(telem[0])), int(float(telem[1])))
					weekday = date.weekday()
					hour = date.hour

					previousweekday = datetime(int(float(day[-20].split('.')[0])), int(float(day[-20].split('.')[1])),
					                           int(float(day[-20].split('.')[2])), int(float(time[-20].split(':')[0])),
					                           int(float(time[-20].split(':')[1]))).weekday()

					sma100 = np.mean(bars[-100:])

					if (weekday == 6):
						if (previousweekday == 6):
							sma = np.mean(bars[-self.window:])
							std = np.std(bars[-self.window:])
							bollinger = (bars[len(bars) - 1] - sma) / (2 * std)
						else:
							bollinger = 0
							print('FIRST 20 MINUTES of the week')

					else:
						sma = np.mean(bars[-self.window:])
						std = np.std(bars[-self.window:])
						bollinger = (bars[len(bars) - 1] - sma) / (2 * std)

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
					elif bollinger < -1.0 and sma100 - 0.0005 > bars[-1] and self.bought[s][0] == 'OUT':
						print("LONG: %s" % bar_date)
						sig_dir = 'LONG'

						signal = SignalEvent(1, symbol, dt, sig_dir, 1.0)
						self.events.put(signal)
						self.bought[s] = ('LONG', bars[-1])

					elif bollinger > 1.0 and sma100 + 0.0005 < bars[-1] and self.bought[s][0] == 'OUT':
						print("SHORT: %s" % bar_date)
						sig_dir = 'SHORT'

						signal = SignalEvent(1, symbol, dt, sig_dir, 1.0)
						self.events.put(signal)
						self.bought[s] = ('SHORT', bars[-1])


					elif bollinger > -0.3 and self.bought[s][0] == 'LONG':
						print("CLOSE POSITION: %s" % bar_date)
						sig_dir = 'EXIT'
						signal = SignalEvent(1, symbol, dt, sig_dir, 1.0)
						self.events.put(signal)
						self.bought[s] = ('OUT', 0)

					elif bollinger < 0.3 and self.bought[s][0] == 'SHORT':
						print("CLOSE POSITION: %s" % bar_date)
						sig_dir = 'EXIT'
						signal = SignalEvent(1, symbol, dt, sig_dir, 1.0)
						self.events.put(signal)
						self.bought[s] = ('OUT', 0)


class PrimeCrossStrategy(Strategy):
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

	def calculate_signals(self, event):

		if event.type == 'MARKET':
			for s in self.symbol_list:
				bars = self.bars.get_latest_bars_values(
					s, "Close", N=self.long_window
				)
				bar_date = self.bars.get_latest_bar_datetime(s)
				if bars is not None:
					short_sma = np.mean(bars[-self.short_window:])
					long_sma = np.mean(bars[-self.long_window:])
					symbol = s
					dt = bar_date
					sig_dir = ""
					print("self_bought", self.bought[s], s)
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
