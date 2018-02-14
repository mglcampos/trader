import unittest2
import re

from htr.core.brokerage import KrakenHandler

class TestKrakenHandler(unittest2.TestCase):
	__test__ = False

	def setUp(self):

		self.crypto_handler = KrakenHandler('')

	## todo create order and test this
	def test_orders(self):
		kraken = KrakenHandler('')

		print("Equity: ", kraken.get_equity())
		print("Portfolio: ", kraken.get_available_units())
		print("XRP :", kraken.get_available_units('XRP'))
		print("USD : ", kraken.get_cash())
		max = kraken.get_max_buy('XRPUSD')
		print('Max amount XRP: ', max)
		print("Downsize amount: ", kraken.downsize_order(max))
		# try:
		# 	print(kraken.create_order('XRPUSD', kraken.get_max_buy('XRPUSD'), 1, 'buy', 'market'))
		#
		# except Exception as e:
		# 	if re.search('Insufficient funds', e.__str__()) != None:
		# 		print(kraken.create_order('XRPUSD', kraken.downsize_order(kraken.get_max_buy('XRPUSD')), 1, 'buy', 'market'))
		# 		
		# print(kraken.create_order('XRPUSD', kraken.get_max_sell('XRPUSD'), 1, 'sell', 'market'))
		#
		print('Max amount XRP: ', kraken.get_max_buy('XRPUSD'))
		print("Equity: ", kraken.get_equity())
		print("Portfolio: ", kraken.get_available_units())
		print("XRP :", kraken.get_available_units('XRP'))
		print("USD : ", kraken.get_cash())

	def tearDown(self):
		pass


if __name__ == '__main__':
	unittest2.main()
