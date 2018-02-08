import unittest2

from htr.core.brokerage import BitfinexHandler


class TestBitfinexHandler(unittest2.TestCase):
	__test__ = True

	def setUp(self):

		exchanges = []
		self.crypto_handler = BitfinexHandler(exchanges)

	# def test_ticker(self):
	#
	# 	print(self.crypto_handler.get_tick('btcusd'))
	# 	print(self.crypto_handler.get_tick('ethusd'))
	# 	print(self.crypto_handler.get_tick('xrpusd'))
	#
	# def test_get_symbols(self):
	#
	# 	print(self.crypto_handler.get_symbols())

	# def test_get_available_cash(self):
	#
	# 	print(self.crypto_handler.get_available_cash('eth'))

	# def test_get_equity(self):
	#
	# 	print(self.crypto_handler.get_equity())

	# def test_pending_orders(self):
	#
	# 	print(self.crypto_handler.get_pending_orders())

	## todo create order and test this
	def test_orders(self):

		print('OPEN POS BEFORE: ', self.crypto_handler.get_open_positions())

		print(self.crypto_handler.create_order('btcusd', 0.003, 1, 'buy', 'exchange market'))
		open_pos = self.crypto_handler.get_open_positions()
		print('OPEN POS BOUGHT: ', open_pos)
		cash = self.crypto_handler.get_available_units(symbol='btc')
		print(self.crypto_handler.create_order('btcusd', cash, 1, 'sell', 'exchange market'))
		print('OPEN POS SOLD: ', self.crypto_handler.get_open_positions())
		print(self.crypto_handler.status_order())
		"""{'side': 'sell', 'cid': 84825908488, 'exchange': 'bitfinex', 'order_id': 7249505476, 'src': 'api', 'was_forced': False, 'gid': None, 'is_hidden': False, 'timestamp': '1516145625.978178984', 'avg_execution_price': '0.0', 'oco_order': None, 'id': 7249505476, 'cid_date': '2018-01-16', 'remaining_amount': '0.003992', 'is_cancelled': False, 'executed_amount': '0.0', """
		"""{'side': 'buy', 'cid': 84824690599, 'exchange': 'bitfinex', 'order_id': 7249504949, 'src': 'api', 'was_forced': False, 'gid': None, 'is_hidden': False, 'timestamp': '1516145624.712958717', 'avg_execution_price': '0.0', 'oco_order': None, 'id': 7249504949, 'cid_date': '2018-01-16', 'remaining_amount': '0.002', 'is_cancelled': False, 'executed_amount': '0.0', """

	def tearDown(self):
		pass


if __name__ == '__main__':
	unittest2.main()
