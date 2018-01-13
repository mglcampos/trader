
import unittest2
import os
import pandas as pd
from htr.helpers.dataprep.dataprep import DataPrep

class TestDataprep(unittest2.TestCase):

    __test__ = True

    def setUp(self):
        self.prep = DataPrep()


    def test_load(self):
        dframes = self.prep.load_csv(os.path.abspath('../../histdata/EURNZD_H1_2012'))
        self.assertEqual(1,len(dframes))
        self.assertIsInstance(dframes[0], pd.DataFrame)

    def test_dataprep(self):
        dframes = self.prep.load_csv(os.path.abspath('../../histdata/EURNZD_H1_2012'))
        dframes = self.prep.prepare(dframes)
        self.assertIsInstance(dframes[0].index, pd.DatetimeIndex)
        print(dframes[0].index.freqstr)

    def tearDown(self):
        pass

if __name__ == '__main__':
    unittest2.main()