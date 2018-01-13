from abc import ABC, abstractmethod


class Portfolio(ABC):
	def __init__(self, data_handler, events, context):

		self.store = context.store
		self.granularity = context.timeframe
		self.data_handler = data_handler
		self.events = events
		self.initial_capital = context.initial_capital

		self.symbol_list = self.data_handler.symbol_list
		self.start_date = self.data_handler.get_start_date(self.symbol_list[0])

		self.all_positions = self.construct_all_positions()
		self.current_positions = dict((k, v) for k, v in \
		                              [(s, 0) for s in self.symbol_list])
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

	def construct_current_holdings(self):
		"""
		This constructs the dictionary which will hold the instantaneous
		value of the portfolio across all symbols.
		"""

		d = dict((k, v) for k, v in [(s, 0.0) for s in self.symbol_list])
		d['cash'] = self.initial_capital
		d['commission'] = 0.0
		d['total'] = self.initial_capital
		return d

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
		dp = dict((k, v) for k, v in [(s, 0) for s in self.symbol_list])
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
			market_value = self.current_positions[s] * \
			               self.data_handler.get_latest_bar_value(s, "Close")

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
		## todo in the crypto case should commission be doubled?
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
