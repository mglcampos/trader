
import unittest2

from htr.core.strategy import StrategyManager


class TestStrategyManager(unittest2.TestCase):

    __test__ = True

    def setUp(self):
        self.strat_manager = StrategyManager()

    def test_strategy_manager(self):

        _class = StrategyManager.get_strategy('mean reverting')

    def tearDown(self):
        pass

if __name__ == '__main__':
    unittest2.main()