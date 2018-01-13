import unittest2

from htr.core.configuration import ConfigManager, Context
from htr.core.data.handler import CsvDataHandler
from htr.core.engines import Backtest
from htr.core.execution import SimulatedExecutionHandler
from htr.core.portfolio import BacktestPortfolio
from htr.core.portfolio.risk import Kelly
from htr.core.strategy import StrategyManager


class TestBacktestEngine(unittest2.TestCase):

	__test__ = True

	def setUp(self):
		self.context = ConfigManager('config.json').get_context()

	# handler, execution_handler, portfolio, strategy, risk_handler, context

	def test_backtest(self):

		merged_dict = dict(list(self.context.backtest_nodes[0].items()) + list(getattr(self.context, Context.SPECS).items()))
		node_context = Context()
		node_context.__dict__ = merged_dict
		self.backtest = Backtest(CsvDataHandler,
		                         SimulatedExecutionHandler,
		                         BacktestPortfolio,
		                         StrategyManager,
		                         Kelly,
		                         node_context).simulate_trading()

	def tearDown(self):
		pass


if __name__ == '__main__':
	unittest2.main()
