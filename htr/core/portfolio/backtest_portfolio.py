

try:
	import Queue as queue
except ImportError:
	import queue
import statsmodels.tsa.stattools as ts

import datetime
from math import floor
# import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from htr.core.events import *
from htr.core.performance import create_sharpe_ratio, create_drawdowns

from pymongo import MongoClient
from talib.abstract import *
import json
from htr.core.portfolio.portfolio import Portfolio
from htr.helpers.stationarity import hurst


class BacktestPortfolio(Portfolio):
	"""Manages the Portfolio in a backtest/simulation scenario."""

	def __init__(self, context, events, risk_handler, data_handler):

		super().__init__(data_handler, events, context)
		self.context = context
		self.events = events
		self.risk_handler = risk_handler
		try:
			self.notes = self.context.notes
		except:
			self.notes = 'N/A'

	def process_signal(self, event):
		"""
		Simply files an Order object as a constant quantity
		sizing of the signal object, without risk management or
		position sizing considerations.
		Parameters:
		signal - The tuple containing Signal information.
		"""

		if event.type == 'SIGNAL':
			quantity = self.risk_handler.calculate_trade(self.current_positions, event, self.data_handler.get_latest_bar_value(event.symbol, "Close"))
			if quantity != 0:
				orders = self._singular_order(event, quantity)

		elif len(event.signals) == 2 and event.type == 'GROUP_SIGNAL':
			## todo fix this
			quantity = self.risk_handler.evaluate_group_trade(self.current_positions, event.signals)
			if quantity != 0:
				orders = self._pair_order(event.signals, quantity)

		elif len(event.signals) == 3 :
			quantity = self.risk_handler.evaluate_group_trade(self.current_positions, event.signals)
			if quantity != 0:
				orders = self._triangular_order(event.signals, quantity)
			##todo when should triangular orders should be used
			pass

		else:
			pass

		for order in orders:
			self.events.put(order)
		#     if signal_type == 'LONG' and cur_quantity == 0:
        #
        #         order_type = 'MKT-OPEN'
        #         self.len_buy += 1
        #         order = OrderEvent(symbol, order_type, mkt_quantity, 'BUY', timestamp=timestamp)
        #
        #     if signal_type == 'SHORT' and cur_quantity == 0:
        #
        #         order_type = 'MKT-OPEN'
        #         self.len_sell += 1
        #
        #         order = OrderEvent(symbol, order_type, mkt_quantity, 'SELL', timestamp=timestamp)
        #
        #     if signal_type == 'EXIT' and cur_quantity > 0:
        #         order_type = 'MKT-CLOSE'
        #
        #
        #         order = OrderEvent(symbol, order_type, abs(cur_quantity), 'SELL', timestamp=timestamp)
        #         print 'Exiting position of - ', symbol
        #
        #     if signal_type == 'EXIT' and cur_quantity < 0:
        #         order_type = 'MKT-CLOSE'
        #
        #
        #         order = OrderEvent(symbol, order_type, abs(cur_quantity), 'BUY', timestamp=timestamp)
        #         print 'Exiting position of - ', symbol
        # # todo contador de longs e de sells, e usar market value para profits



	def _pair_order(self, signals, quantity):
		"""."""

		mkt_quantity = quantity/2
		orders = []
		for signal in signals:
			signal_type = signal.signal_type
			strength = signal.strength
			timestamp = signal.datetime
			symbol = signal.symbol
			cur_quantity = self.current_positions[symbol]


			if signal_type == 'LONG' and cur_quantity == 0:
				market_value = mkt_quantity * self.data_handler.get_latest_bar_value(self.symbol_list[0], "Close")
				market_value = market_value - (
				mkt_quantity * self.data_handler.get_latest_bar_value(self.symbol_list[1], "Close"))  ##valor da moeda
				self.last_open = self.last_market  ##cash antes do open
				self.cash_open = self.last_market - market_value  ##somar valor da moeda mais cash

				order_type = 'MKT-OPEN'
				self.len_buy += 1
				orders.append(OrderEvent(symbol, order_type, mkt_quantity, 'BUY', timestamp=timestamp))

			if signal_type == 'SHORT' and cur_quantity == 0:
				market_value = - mkt_quantity * self.data_handler.get_latest_bar_value(self.symbol_list[0], "Close")
				market_value = market_value + (
				mkt_quantity * self.data_handler.get_latest_bar_value(self.symbol_list[1], "Close"))  ##valor da moeda
				self.last_open = self.last_market  ##cash antes do open
				self.cash_open = self.last_market + market_value  ##somar valor da moeda mais cash

				order_type = 'MKT-OPEN'
				self.len_sell += 1

				orders.append(OrderEvent(symbol, order_type, mkt_quantity, 'SELL', timestamp=timestamp))

			if signal_type == 'EXIT' and cur_quantity > 0:
				order_type = 'MKT-CLOSE'
				print('timestamp', timestamp)
				print('len_symbols', len(self.symbol_list))
				print('current_holdings', self.current_holdings['cash'])
				print('last_open', self.last_open)
				print('cash_open', self.cash_open)
				cash = (cur_quantity * self.data_handler.get_latest_bar_value(self.symbol_list[0],
				                                                      "Close")) + self.current_holdings[
					       'cash']  ##tirar valor da moeda ao cash
				cash = cash - (cur_quantity * self.data_handler.get_latest_bar_value(self.symbol_list[1],
				                                                             "Close"))
				market_value = cash - self.last_open
				print('market_value', market_value)
				print('cash', cash)
				if market_value > 0:
					self.profit_buy += market_value
					self.consecutive_win += 1
					self.loss_streak.append(self.consecutive_loss)
					self.consecutive_loss = 0
				else:
					self.loss_buy += market_value
					self.win_streak.append(self.consecutive_win)
					self.consecutive_win = 0
					self.consecutive_loss += 1

				orders.append(OrderEvent(symbol, order_type, abs(cur_quantity), 'SELL', timestamp=timestamp))
				print('Exiting position of - ', symbol)

			if signal_type == 'EXIT' and cur_quantity < 0:
				order_type = 'MKT-CLOSE'
				print('timestamp', timestamp)
				print('last_open', self.last_open)
				print('current_holdings', self.current_holdings['cash'])
				print('cash_open', self.cash_open)
				print(self.symbol_list[0], (cur_quantity * self.data_handler.get_latest_bar_value(self.symbol_list[0],
				                                                                          "Close")))
				cash = self.current_holdings['cash'] + (cur_quantity * self.data_handler.get_latest_bar_value(self.symbol_list[0],
				                                                                                      "Close"))  ##tirar valor da moeda ao cash
				print('1stcash', cash)
				cash = cash + (-cur_quantity * self.data_handler.get_latest_bar_value(self.symbol_list[1],
				                                                              "Close"))
				market_value = cash - self.last_open
				print('market_value', market_value)
				print('2ndcash', cash)
				if market_value > 0:
					self.profit_sell += market_value
					self.consecutive_win += 1
					self.loss_streak.append(self.consecutive_loss)
					self.consecutive_loss = 0
				else:
					self.loss_sell += market_value
					self.win_streak.append(self.consecutive_win)
					self.consecutive_win = 0
					self.consecutive_loss += 1

				orders.append(OrderEvent(symbol, order_type, abs(cur_quantity), 'BUY', timestamp=timestamp))
				print('Exiting position of - ', symbol)

		return orders


	def _singular_order(self, signal, quantity):
		"""."""

		signal_type = signal.signal_type
		strength = signal.strength
		timestamp = signal.datetime
		symbol = signal.symbol
		cur_quantity = self.current_positions[symbol]
		available_cash = self.last_market - (quantity * self.data_handler.get_latest_bar_value(signal.symbol, "Close") * self.context.commission)

		if signal_type == 'LONG':

			if quantity > available_cash / self.data_handler.get_latest_bar_value(signal.symbol, "Close"):
				mkt_quantity = available_cash / self.data_handler.get_latest_bar_value(signal.symbol, "Close")
			else:
				mkt_quantity = quantity
			print('\n AVAILABLE CASH: ', quantity, available_cash, mkt_quantity  )
			market_value = mkt_quantity * self.data_handler.get_latest_bar_value(symbol, "Close")  ##valor da moeda
			self.last_open = self.last_market  ##cash antes do open
			self.cash_open = self.last_market - market_value  ##somar valor da moeda mais cash
	
			order_type = 'MKT-OPEN'
			self.len_buy += 1
			order = OrderEvent(symbol, order_type, mkt_quantity, 'BUY', timestamp=timestamp)
	
		elif signal_type == 'SHORT':

			if quantity < available_cash:
				mkt_quantity = available_cash
			else:
				mkt_quantity = quantity

			market_value = quantity * self.data_handler.get_latest_bar_value(symbol, "Close")  ##valor da moeda
			self.last_open = self.last_market  ##cash antes do open
			self.cash_open = self.last_market + market_value  ##somar valor da moeda mais cash
	
			order_type = 'MKT-OPEN'
			self.len_sell += 1
	
			order = OrderEvent(symbol, order_type, quantity, 'SELL', timestamp=timestamp)
	
		elif signal_type == 'EXIT' and cur_quantity > 0:

			if cur_quantity >= quantity:
				mkt_quantity = quantity
			else:
				mkt_quantity = cur_quantity

			order_type = 'MKT-CLOSE'
			print('timestamp', timestamp)
			print('last_open', self.last_open)
			print('current_holdings', self.current_holdings['cash'])
			print('cash_open', self.cash_open)
			cash = cur_quantity * self.data_handler.get_latest_bar_value(symbol,
			                                                     "Close") + self.cash_open  ##tirar valor da moeda ao cash
			market_value = cash - self.last_open
			print('market_value', market_value)
			print('cash', cash)
			if market_value > 0:
				self.profit_buy += market_value
				self.consecutive_win += 1
				self.loss_streak.append(self.consecutive_loss)
				self.consecutive_loss = 0
			else:
				self.loss_buy += market_value
				self.win_streak.append(self.consecutive_win)
				self.consecutive_win = 0
				self.consecutive_loss += 1
	
			order = OrderEvent(symbol, order_type, abs(cur_quantity), 'SELL', timestamp=timestamp)
			print('Exiting position of - ', symbol)
	
		elif signal_type == 'EXIT' and cur_quantity < 0:

			if cur_quantity <= quantity:
				mkt_quantity = quantity
			else:
				mkt_quantity = cur_quantity

			order_type = 'MKT-CLOSE'
			print('timestamp', timestamp)
			print('current_holdings', self.current_holdings['cash'])
			print('last_open', self.last_open)
			print('cash_open', self.cash_open)
			cash = mkt_quantity * self.data_handler.get_latest_bar_value(symbol,
			                                                     "Close") + self.cash_open  ##tirar valor da moeda ao cash
			market_value = cash - self.last_open
			print('market_value', market_value)
			print('cash', cash)
			if market_value > 0:
				self.profit_sell += market_value
				self.consecutive_win += 1
				self.loss_streak.append(self.consecutive_loss)
				self.consecutive_loss = 0
			else:
				self.loss_sell += market_value
				self.win_streak.append(self.consecutive_win)
				self.consecutive_win = 0
				self.consecutive_loss += 1
	
			order = OrderEvent(symbol, order_type, abs(mkt_quantity), 'BUY', timestamp=timestamp)
			print('Exiting position of - ', symbol)

		else:
			raise ValueError('Empty order error for signal {} with current quantity of {}.'.format(signal, cur_quantity))
		
		return [order]

	def tringular_order(self, signal):

		raise NotImplementedError('Should implement')
	
	def _enter_position(self, signal):
		pass
		
	def _exit_position(self, signal):
		pass

	def construct_all_positions(self):
		"""
		For each symbol 0 positions
		Constructs the positions list using the start_date
		to determine when the time index will begin.
		"""

		d = dict((k, v) for k, v in [(s, 0) for s in self.symbol_list])
		d['datetime'] = self.start_date
		return [d]

	def construct_all_holdings(self):
		"""
		For each symbol 0 in holdings
		Constructs the holdings list using the start_date
		to determine when the time index will begin.
		"""

		d = dict((k, v) for k, v in [(s, 0.0) for s in self.symbol_list])
		d['datetime'] = self.start_date
		d['cash'] = self.initial_capital
		d['commission'] = 0.0
		d['total'] = self.initial_capital
		return [d]
	
	def create_equity_curve_dataframe(self):
		"""
		Creates a pandas DataFrame from the all_holdings
		list of dictionaries.
		"""

		curve = pd.DataFrame(self.all_holdings)
		print(curve)
		curve.set_index('datetime', inplace=True)
		curve['returns'] = curve['total'].pct_change()
		curve['equity_curve'] = (1.0 + curve['returns']).cumprod()
		# self.equity_curve = curve.dropna(axis=0, how='any')
		self.equity_curve = curve
	
	def output_summary_stats(self, symbol_data):
		"""
		Creates a list of summary statistics for the portfolio.
		"""
		# if period == 'M1' or period == 'M5' or period == 'M15':
		#     periods = 252*6.5*60
		# elif period == 'H1' or period == 'H4':
		#     periods = 252*6.5
		# else:
		#     periods = 252
		periods = len(symbol_data[self.symbol_list[0]]) # Nr of entries in the dataset, used to calculate the sharpe ratio for instance
		files = 'N/A'
		if 'path'in self.context.data_sources[0].keys():
			files = self.context.data_sources

		# self.equity_curve = self.equity_curve.dropna(axis=0)
		total_return = self.equity_curve['equity_curve'][-1]
		returns = self.equity_curve['returns']
		pnl = self.equity_curve['equity_curve']
		pnl = pnl.dropna()
		sharpe_ratio = create_sharpe_ratio(returns, periods=periods)
		drawdown, max_dd, dd_duration = create_drawdowns(pnl)
		self.equity_curve['drawdown'] = drawdown

	
		l = len(symbol_data) * 100
		i = l + 11
		markers = {'BUY': {}, 'SELL': {}, 'CLOSE': {}}
		flags = {'BUY': {}, 'SELL': {}, 'CLOSE': {}}
		volatility = {}
		for instrument in symbol_data:
			if 'Open' in symbol_data[instrument].keys():
				inputs = {
					'open': symbol_data[instrument]['Open'],
					'high': symbol_data[instrument]['High'],
					'low': symbol_data[instrument]['Low'],
					'close': symbol_data[instrument]['Close']
				}
				# try:
				# 	pass
				# 	# atr = ATR(inputs, timeperiod=len(symbol_data[instrument]) - 1)
				# 	# volatility[instrument] = (atr[-1] / symbol_data[instrument]['Close'].values[-1]) * 100
				# 	l = 0
				# 	hurst100 = []
				# 	adf100 = []
				# 	while l < len(symbol_data[instrument]['Close']):
				# 		hurst100.append(hurst(symbol_data[instrument]['Close'].iloc[l:l+100]))
				# 		adf100.append(ts.adfuller(symbol_data[instrument]['Close'].iloc[l:l+100])[0])
				# 		l += 100
				#
				# 	# Lets plot
				# 	fig = plt.figure(1)
				# 	fig.suptitle('Performance', fontsize=16)
				# 	ax = plt.subplot(511)
				# 	ax.title.set_text('Equity Curve')
				# 	self.equity_curve['equity_curve'].plot(legend=None)
				# 	ax = plt.subplot(512)
				# 	ax.title.set_text('ADF')
				# 	atr = pd.Series(adf100)
				# 	atr.plot(legend=None)
				# 	fig.subplots_adjust(hspace=1)
				# 	ax = plt.subplot(513)
				# 	ax.title.set_text('Hurst')
				# 	adx = pd.Series(hurst100)
				# 	adx.plot(legend=None)
				# 	fig.subplots_adjust(hspace=1)
				# 	ax = plt.subplot(514)
				# 	ax.title.set_text('RSI')
				# 	rsi = pd.Series(RSI(inputs, timeperiod=21))
				# 	rsi.plot(legend=None)
				# 	sd = symbol_data[instrument]['Close']
				# 	fig.subplots_adjust(hspace=1)
				# 	ax = plt.subplot(515)
				# 	ax.title.set_text(instrument)
				# 	sd = pd.Series(sd)
				# 	sd.plot(legend=None)
				# 	plt.show()
				#
				# except Exception as e:
				# 	volatility[instrument] = 0
				# 	print("COULDN'T calculate volatility", e)
			#
			# fig2 = plt.figure(2)
			# fig2.suptitle('Instruments', fontsize=16)
			# ax2 = plt.subplot(i)
			# ax2.title.set_text(instrument)
			# print(symbol_data[instrument])
			symbol_data[instrument] = symbol_data[instrument].dropna(axis=0, how='any')
			# print(symbol_data[instrument])
			for key in markers:
				markers[key][instrument] = []
				flags[key][instrument] = {}
				if instrument in self.fill_history[key]:
					flags[key][instrument]['X'] = []
					flags[key][instrument]['Title'] = []
					flags[key][instrument]['Text'] = []
					for index in self.fill_history[key][instrument]:
						# print("row nr:", np.where(symbol_data[instrument].index == index))
						if np.where(symbol_data[instrument].index == index)[0]:
							# markers[key][instrument].append(np.where(symbol_data[instrument].index == index)[0][0])
							try:
								markers[key][instrument].append(
									symbol_data[instrument]['Close'][np.where(symbol_data[instrument].index == index)[0][0]])
							except KeyError:
								markers[key][instrument].append(
									symbol_data[instrument][
										np.where(symbol_data[instrument].index == index)[0][0]])
							flags[key][instrument]['X'].append(index)
							flags[key][instrument]['Title'].append(key)
							flags[key][instrument]['Text'].append(instrument)
	
	
	
							##TODO symbol_data column is hardcoded to Close
	
		report = pd.DataFrame()
		buy = pd.DataFrame({'Buy': pd.Series(None), 'Buy_Y': pd.Series(None)})
		sell = pd.DataFrame({'Sell': pd.Series(None), 'Sell_Y': pd.Series(None)})
		close = pd.DataFrame({'Close': pd.Series(None), 'Close_Y': pd.Series(None)})
	
		# TODO if len(symbol_list == 2) volatility = [x,y]  stddev(returns) or atr/close

		for instrument in symbol_data:
			if 'X' in flags['BUY'][instrument] and 'BUY' in markers:
				buy = pd.DataFrame({'Buy': flags['BUY'][instrument]['X'],
				                    'Buy_Y': markers['BUY'][instrument]})  ## keys - instrument todo change this
			if 'X' in flags['SELL'][instrument] and 'SELL' in markers:
				sell = pd.DataFrame({'Sell': flags['SELL'][instrument]['X'], 'Sell_Y': markers['SELL'][instrument]})
			if 'X' in flags['CLOSE'][instrument] and 'CLOSE' in markers:
				close = pd.DataFrame({'Close': flags['CLOSE'][instrument]['X'], 'Close_Y': markers['CLOSE'][instrument]})
	
			report = pd.concat([buy, sell, close], ignore_index=True, axis=1)
			# print(report_builder.head())
			# print('COLUNAS', report_builder.head())
			s = str(instrument).replace("/", "")
			report.columns = ['Buy', 'Buy_Y', 'Sell', 'Sell_Y', 'Close', 'Close_Y']
			report.to_csv(s + '_report.csv', columns=['Buy', 'Buy_Y', 'Sell', 'Sell_Y', 'Close', 'Close_Y'], header=False)
	
		# sharpe = pd.DataFrame({'Sharpe_Ratio': pd.Series(sharpe_ratio)})
		# max_drawdown = pd.DataFrame({'Max_Drawdown': pd.Series(max_dd * 100.0)})
		# drawdown_duration = pd.DataFrame({'Drawdown_Duration': pd.Series(dd_duration)})
		consecutive_wins = 0
		consecutive_losses = 0
		if len(self.win_streak) > 1:
			consecutive_wins = max(self.win_streak)

		if len(self.loss_streak) > 1:
			consecutive_losses = max(self.loss_streak)

		reportf = {
			'name': self.context.name,
			'files': files,
			'strategy': self.context.strategies,
			'header': self.context.data_header,
			'instruments': self.symbol_list,
			'sharpe': float(sharpe_ratio),
			'max_drawdown': float(max_dd * 500.0),
			'drawdown_duration': float(dd_duration),
			'len_bars': int(symbol_data[self.symbol_list[0]].shape[0]),
			'len_sell': int(self.len_sell),
			'len_buy': int(self.len_buy),
			'profit': float((total_return - 1.0) * 100.0),
			'profit_sell': float(self.profit_sell),
			'profit_buy': float(self.profit_buy),
			'loss_sell': float(self.loss_sell),
			'loss_buy': float(self.loss_buy),
			'consecutive_wins': int(consecutive_wins),
			'consecutive_losses': int(consecutive_losses),
			'volatility': volatility,
			'granularity': self.context.timeframe,
			'notes': self.notes
		}
		with open('reportf.json', 'w') as outfile:
			json.dump(reportf, outfile)
		if self.store == True:
			client = MongoClient('localhost', 27017)
			results = client['backtests']
			bresults = results.backtest
			post_id = bresults.insert_one(reportf).inserted_id
			print('id inserido', post_id)
	
		# print('COLUNAS',report_builder.columns.values)
		print(report.head())
	
		stats = [("Total Return", "%0.2f%%" % \
		          ((total_return - 1.0) * 100.0)),
		         ("Sharpe Ratio", "%0.2f" % sharpe_ratio),
		         ("Max Drawdown", "%0.2f%%" % (max_dd * 100.0)),
		         ("Drawdown Duration", "%d" % dd_duration)]
	
		self.equity_curve.to_csv('equity.csv')
		return stats
