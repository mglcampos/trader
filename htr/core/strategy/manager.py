
from importlib.machinery import SourceFileLoader
import os

from htr.helpers.class_finder import FilePaths, get_class

class StrategyManager:
	"""Manages all the strategies, instantiates them and calls them to generate trading signals."""

	def __init__(self, context, events, data_handler):

		self.context = context
		self.events = events
		self.data_handler = data_handler
		self.strategies = []
		self.symbol_list = self.data_handler.symbol_list
		# self.bought = self._calculate_initial_bought()

		##todo add passing params dict via context, e.g. strategies = [{strategy:'a', params:{}}]
		for strategy in context.strategies:
			strategy_cls = get_class(strategy, FilePaths.STRATEGY)
			# Instantiates strategy classes.
			self.strategies.append(strategy_cls(self.context, self.events, self.data_handler))

	def calculate_signals(self):
		"""Calls all strategies to gather signal events shared via events queue."""
		## todo currently is creating 1 position per strategy
		for strategy in self.strategies:
			strategy.calculate_signals()

	## todo solve the bought issue, common repo for fill positions.
	# def _calculate_initial_bought(self):
	# 	"""Cache where open positions and the price when they were opened/close are stored for strategy use."""
	#
	# 	bought = {}
	# 	for s in self.symbol_list:
	# 		bought[s] = ('OUT', 0)
	#
	# 	return bought
	#
	# def set_bought(self, symbol, status, price):
	# 	"""Sets the current position status and price."""
	#
	# 	self.bought[symbol] = (status, price)
	#
	# def get_bought(self, symbol):
	# 	"""Returns the the current position status and price."""
	#
	# 	return self.bought[symbol]


