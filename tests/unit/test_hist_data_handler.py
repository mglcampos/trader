import queue

import unittest2

from htr.core.configuration import ConfigManager
from htr.core.data.handler import CsvDataHandler


class TestHistDataHandler(unittest2.TestCase):

    __test__ = True

    def setUp(self):

        self.context = ConfigManager('config.json').get_context()
        self.events = queue.Queue()


    def test_init(self):

        self.dhandler = CsvDataHandler(self.context, self.events)
        self.assertEqual(1, len(self.dhandler.dframes['EUR/AUD']))

    def test_get_dframes(self):

        self.dhandler = CsvDataHandler(self.context, self.events)
        self.dhandler.update_bars()
        print(self.dhandler.get_latest_bar('EUR/AUD'))

    def tearDown(self):
        pass

if __name__ == '__main__':
    unittest2.main()