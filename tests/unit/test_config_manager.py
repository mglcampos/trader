
import unittest2

from htr.core.configuration import ConfigManager

class TestConfigManager(unittest2.TestCase):

    __test__ = True

    def setUp(self):
        self.cmanager = ConfigManager('config.json')

    def test_config_manager(self):
        # print(self.cmanager.context)
        # for k in self.cmanager.context.__dict__.keys():
        #     self.assertIsInstance(getattr(self.cmanager.context, k), dict)
        #     self.assertGreater(len(getattr(self.cmanager.context, k).keys()), 0)
        pass
    def tearDown(self):
        pass

if __name__ == '__main__':
    unittest2.main()