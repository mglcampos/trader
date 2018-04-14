from abc import ABCMeta, abstractmethod


class BrokerHandler(object):
	"""

	"""

	__metaclass__ = ABCMeta

	@abstractmethod
	def create_order(self, symbol, amount, price, side, ord_type):
		"""Creates an Limit/Market/Stop order."""

		pass

	@abstractmethod
	def get_tick(self, symbol):
		"""Retrieves last tick."""

		pass

	@abstractmethod
	def get_equity(self):
		"""Returns all the equity."""

		pass

	@abstractmethod
	def get_cash(self):
		"""Returns the available cash."""

		pass

