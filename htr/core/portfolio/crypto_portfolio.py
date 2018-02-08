

from datetime import datetime as dt
import time

try:
	import Queue as queue
except ImportError:
	import queue

# from pymongo import MongoClient
from talib.abstract import *
import json

from htr.core.events import *
from .portfolio import Portfolio


class CryptoPortfolio():
	"""Manages the Portfolio in a crypto currency live trading scenario."""

	def __init__(self, context, events, risk_handler, data_handler, broker_handler):

		self.store = context.store
		self.context = context
		self.data_handler = data_handler
		self.events = events
		self.broker_handler = broker_handler
		self.initial_capital = context.initial_capital

		self.symbol_list = self.data_handler.symbol_list
		self.start_date = dt.now()

		self.current_positions = dict((k, v) for k, v in \
		                              [(s, 0) for s in self.symbol_list])
		## Update current positions
		self.update_pos_from_broker()

		self.all_positions = self.construct_all_positions()


		self.all_holdings = self.construct_all_holdings()
		self.current_holdings = self.construct_current_holdings()
		self.fill_history = {'BUY': {}, 'SELL': {}, 'CLOSE': {}}
		##todo check if all variables are being used, maybe use a dict to aggr
		self.len_sell = 0
		self.len_buy = 0
		self.profit_buy = 0
		self.profit_sell = 0
		self.consecutive_win = 0
		self.loss_streak = []
		self.win_streak = []
		self.consecutive_loss = 0
		self.last_open = 0
		self.loss_buy = 0
		self.loss_sell = 0
		self.last_market = 0

		self.events = events
		self.risk_handler = risk_handler
		self.broker_handler = broker_handler

	def process_signal(self, event):
		"""
		Simply files an Order object as a constant quantity
		sizing of the signal object, without risk management or
		position sizing considerations.
		Parameters:
		signal - The tuple containing Signal information.
		"""

		order = None

		if event.type == 'SIGNAL':
			quantity = self.risk_handler.calculate_trade(self.current_positions, event,
			                                             self.data_handler.get_latest_bar_value(event.symbol, "Close"))
			if quantity != 0:
				order = self._singular_order(event, quantity)

		elif len(event.signals) == 2 and event.type == 'GROUP_SIGNAL':
			## todo fix this
			quantity = self.risk_handler.evaluate_group_trade(self.current_positions, event,
			                                                  self.data_handler.get_latest_bar_value(event.symbol,
			                                                                                         "Close"))
			if quantity != 0:
				order = self._pair_order(event)

		elif len(event.signals) == 3:
			quantity = self.risk_handler.evaluate_group_trade(self.current_positions, event,
			                                                  self.data_handler.get_latest_bar_value(event.symbol,
			                                                                                         "Close"))
			if quantity != 0:
				order = self._triangular_order(event)
			##todo when should triangular orders should be used
			pass

		else:
			pass

		self.events.put(order)

	def _pair_order(self, signal, quantity):
		signal_type = signal.signal_type
		strength = signal.strength
		timestamp = signal.datetime
		symbol = signal.symbol
		cur_quantity = self.current_positions[symbol]
		mkt_quantity = quantity

		if signal_type == 'LONG' and cur_quantity == 0:
			market_value = mkt_quantity * self.data_handler.get_latest_bar_value(self.symbol_list[0], "Close")
			market_value = market_value - (
				mkt_quantity * self.data_handler.get_latest_bar_value(self.symbol_list[1], "Close"))  ##valor da moeda
			self.last_open = self.last_market  ##cash antes do open
			self.cash_open = self.last_market - market_value  ##somar valor da moeda mais cash

			order_type = 'MKT-OPEN'
			self.len_buy += 1
			order = OrderEvent(symbol, order_type, mkt_quantity, 'BUY', timestamp=timestamp)

		if signal_type == 'SHORT' and cur_quantity == 0:
			market_value = - mkt_quantity * self.data_handler.get_latest_bar_value(self.symbol_list[0], "Close")
			market_value = market_value + (
				mkt_quantity * self.data_handler.get_latest_bar_value(self.symbol_list[1], "Close"))  ##valor da moeda
			self.last_open = self.last_market  ##cash antes do open
			self.cash_open = self.last_market + market_value  ##somar valor da moeda mais cash

			order_type = 'MKT-OPEN'
			self.len_sell += 1

			order = OrderEvent(symbol, order_type, mkt_quantity, 'SELL', timestamp=timestamp)

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

			order = OrderEvent(symbol, order_type, abs(cur_quantity), 'SELL', timestamp=timestamp)
			print
			'Exiting position of - ', symbol

		if signal_type == 'EXIT' and cur_quantity < 0:
			order_type = 'MKT-CLOSE'
			print('timestamp', timestamp)
			print('last_open', self.last_open)
			print('current_holdings', self.current_holdings['cash'])
			print('cash_open', self.cash_open)
			print(self.symbol_list[0], (cur_quantity * self.data_handler.get_latest_bar_value(self.symbol_list[0],
			                                                                                  "Close")))
			cash = self.current_holdings['cash'] + (
			cur_quantity * self.data_handler.get_latest_bar_value(self.symbol_list[0],
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

			order = OrderEvent(symbol, order_type, abs(cur_quantity), 'BUY', timestamp=timestamp)
			print
			'Exiting position of - ', symbol

	def _singular_order(self, signal, quantity):

		## todo assert quantity > available quantity
		signal_type = signal.signal_type
		strength = signal.strength
		timestamp = signal.datetime
		symbol = signal.symbol
		cur_quantity = self.current_positions[symbol]
		mkt_quantity = quantity

		if signal_type == 'LONG':
			print(self.current_holdings)
			market_value = mkt_quantity * self.data_handler.get_latest_bar_value(symbol, "Close")  ##valor da moeda
			self.last_open = self.last_market  ##cash antes do open
			self.cash_open = self.last_market - market_value  ##somar valor da moeda mais cash

			order_type = 'MKT-OPEN'
			self.len_buy += 1
			order = OrderEvent(symbol, order_type, mkt_quantity, 'BUY', timestamp=timestamp)

		elif signal_type == 'SHORT':
			market_value = mkt_quantity * self.data_handler.get_latest_bar_value(symbol, "Close")  ##valor da moeda
			self.last_open = self.last_market  ##cash antes do open
			self.cash_open = self.last_market + market_value  ##somar valor da moeda mais cash

			order_type = 'MKT-OPEN'
			self.len_sell += 1

			order = OrderEvent(symbol, order_type, mkt_quantity, 'SELL', timestamp=timestamp)

		elif signal_type == 'EXIT' and cur_quantity > 0:
			order_type = 'MKT-CLOSE'
			print('timestamp', timestamp)
			print('last_open', self.last_open)
			print('current_holdings', self.current_holdings['cash'])
			print('cash_open', self.cash_open)
			cash = mkt_quantity * self.data_handler.get_latest_bar_value(symbol,
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

			order = OrderEvent(symbol, order_type, abs(mkt_quantity), 'SELL', timestamp=timestamp)
			print('Exiting position of - ', symbol)

		elif signal_type == 'EXIT' and cur_quantity < 0:
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
			raise ValueError(
				'Empty order error for signal {} with current quantity of {}.'.format(signal, cur_quantity))

		return order

	def tringular_order(self, signal):

		raise NotImplementedError('Should implement')

	def _enter_position(self, signal):
		pass

	def _exit_position(self, signal):
		pass

	def update_pos_from_broker(self):
		"""."""

		positions = self.broker_handler.get_available_units()
		for s in positions:
			if s[1:] == self.context.base_currency:
				self.current_positions['cash'] = float(positions[s])

			else:
				self.current_positions[s[1:] + '/' + self.context.base_currency] = float(positions[s])

	def construct_all_positions(self):
		"""
		For each symbol 0 positions
		Constructs the positions list using the start_date
		to determine when the time index will begin.
		"""

		d = dict((k, v) for k, v in [(s, self.current_positions[s]) for s in self.symbol_list])

		d['datetime'] = self.start_date
		return [d]

	def construct_current_holdings(self):
		"""
		This constructs the dictionary which will hold the instantaneous
		value of the portfolio across all symbols.
		"""

		d = dict((k, v) for k, v in [(s, self.current_positions[s]) for s in self.symbol_list])

		d['cash'] = self.initial_capital
		d['commission'] = 0.0
		d['total'] = self.initial_capital
		return d

	def construct_all_holdings(self):
		"""
		For each symbol 0 in holdings
		Constructs the holdings list using the start_date
		to determine when the time index will begin.
		"""

		d = dict((k, v) for k, v in [(s, self.current_positions[s]) for s in self.symbol_list])

		d['datetime'] = self.start_date
		d['cash'] = self.initial_capital
		print(d)
		d['commission'] = 0.0 ## todo why do i need this
		d['total'] = self.initial_capital
		return [d]

	def update_timeindex(self, event):
		"""
		Adds a new record to the positions matrix for the current
		market data bar. This reflects the PREVIOUS bar, i.e. all
		current market data at this stage is known (OHLCV).
		Makes use of a MarketEvent from the events queue.
		"""
		self.last_market = self.current_holdings['cash']

		latest_datetime = self.data_handler.get_latest_bar_datetime(
			self.symbol_list[0]
		)
		# Update positions
		# ================
		## todo what to do here when exception occurs?
		while True:
			try:
				positions = self.broker_handler.get_available_units()
				break

			except Exception as e:
				print(e)
				time.sleep(5)
				continue

		dp = dict((k, v) for k, v in [(s, 0) for s in self.symbol_list])

		for s in positions:
			dp[s[1:]] = positions[s]

		dp['datetime'] = latest_datetime
		for s in self.symbol_list:
			dp[s] = self.current_positions[s]
		# Append the current positions
		self.all_positions.append(dp)
		# Update holdings
		# ===============
		dh = dict((k, v) for k, v in [(s, 0) for s in self.symbol_list])
		dh['datetime'] = latest_datetime
		dh['cash'] = self.current_holdings['cash']
		dh['commission'] = self.current_holdings['commission']
		dh['total'] = self.current_holdings['cash']
		for s in self.symbol_list:
			# Approximation to the real value
			market_value = self.current_positions[s] * self.data_handler.get_latest_bar_value(s, "Close")
			dh[s] = market_value
			dh['total'] += market_value

		# Append the current holdings
		self.all_holdings.append(dh)

	def update_positions_from_fill(self, fill):
		"""
		Takes a Fill object and updates the position matrix to
		reflect the new position.
		Parameters:
		fill - The Fill object to update the positions with.
		"""

		# Check whether the fill is a buy or sell
		fill_dir = 0
		if fill.direction == 'BUY':
			fill_dir = 1
		if fill.direction == 'SELL':
			fill_dir = -1
		# Update positions list with new quantities
		self.current_positions[fill.symbol] += fill_dir * fill.quantity

	def update_holdings_from_fill(self, fill):
		"""
		Takes a Fill object and updates the holdings matrix to
		reflect the holdings value.
		Parameters:
		fill - The Fill object to update the holdings with.
		"""

		# Check whether the fill is a buy or sell
		fill_dir = 0
		if fill.direction == 'BUY':
			fill_dir = 1
		if fill.direction == 'SELL':
			fill_dir = -1
		# Update holdings list with new quantities
		fill_cost = self.data_handler.get_latest_bar_value(fill.symbol, "Close")
		cost = fill_dir * fill_cost * fill.quantity
		commission = cost * fill.commission

		self.current_holdings[fill.symbol] += cost
		self.current_holdings['commission'] += commission
		self.current_holdings['cash'] -= (cost + commission)
		self.current_holdings['total'] -= (cost + commission)

	def update_fill(self, event):
		"""
		Updates the portfolio current positions and holdings
		from a FillEvent.
		"""
		print("update_fill")
		if event.type == 'FILL':
			if (event.position == 'MKT-OPEN' and event.direction == 'BUY'):
				action = event.direction

			elif (event.position == 'MKT-OPEN' and event.direction == 'SELL'):
				action = event.direction

			else:
				action = 'CLOSE'

			self.update_positions_from_fill(event)
			print("FILL TIMESTAMP", event.timeindex, event.position, event.direction)
			if event.symbol not in self.fill_history[action].keys():
				self.fill_history[action][event.symbol] = [event.timeindex]
			else:
				self.fill_history[action][event.symbol].append(event.timeindex)

		self.update_holdings_from_fill(event)


