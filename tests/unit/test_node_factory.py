import unittest2

from htr.core.factory import NodeFactory, FactoryType


class TestNodeFactory(unittest2.TestCase):
	__test__ = True

	def setUp(self):
		pass

	def test_strategy_manager(self):
		self.factory = NodeFactory(FactoryType.SEQUENTIAL, True)

	def tearDown(self):
		pass


if __name__ == '__main__':
	unittest2.main()
