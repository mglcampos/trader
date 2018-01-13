import unittest2

from htr.core.data.handler import CsvDataHandler
from htr.core.portfolio import BacktestPortfolio

class TestBacktestPortfolio(unittest2.TestCase):

    __test__ = True

    def setUp(self):

        self.data_handler = CsvDataHandler()
        self.portfolio = BacktestPortfolio()

    def test_generate_order(self):

       pass

    def tearDown(self):
        pass

if __name__ == '__main__':
    unittest2.main()
