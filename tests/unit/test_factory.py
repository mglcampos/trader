import unittest2

from htr.core.factory import NodeFactory, FactoryType

class TestFactory(unittest2.TestCase):

    __test__ = True

    def setUp(self):
        pass

    # def test_sequential_factory(self):
    #     self.factory = NodeFactory(FactoryType.SEQUENTIAL, 'Simulation')

    def test_live_factory(self):
        self.factory = NodeFactory(FactoryType.SEQUENTIAL, 'Live')

    def tearDown(self):
        pass

if __name__ == '__main__':
    unittest2.main()