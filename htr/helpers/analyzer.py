
import datetime
from pymongo import MongoClient
import os

from htr.helpers.dataprep import DataPrep
from htr.helpers.cointegration import study_samples, hurst, adf

class Analyzer:

    def __init__(self, start_date = '2018-01-01', end_date = '2018-04-01'):
        """."""

        # self.afile = '/home/mcampos/Documents/code/trader/histdata/forex/NZDUSD15.csv'
        # self.bfile = '/home/mcampos/Documents/code/trader/histdata/forex/AUDUSD15.csv'
        # self.header = ['Day', 'Time', 'Open', 'High', 'Low', 'Close', 'Volume']
        self.start_date = '2018-01-01'
        self.end_date = '2018-04-01'

    def study_pair(self, afile, bfile, header= ['Day', 'Time', 'Open', 'High', 'Low', 'Close', 'Volume']):
        """."""

        ## Prepare data.
        dt = DataPrep()
        samples = []

        files = [afile.split('/')[-1], bfile.split('/')[-1]]
        print('Files: ', files)
        samples.append(dt.load_csv(afile, header=header)[0])
        samples.append(dt.load_csv(bfile, header=header)[0])
        samples = dt.prepare(samples)
        samples = dt.resize(samples[0][(samples[0].index > self.start_date) & (samples[0].index < self.end_date)],
                            samples[1][(samples[1].index > self.start_date) & (samples[1].index < self.end_date)])

        comb_index = samples[0].index
        comb_index.union(samples[1].index)
        samples = [df.reindex(index=comb_index, method='pad') for df in samples]
        print(samples[0].head())
        print(samples[1].head())

        ## Study data.
        hurst, cadf = study_samples(samples[0], samples[1])

        report = {
            'files': files,
            'start_date': datetime.datetime.strptime(samples[0]['Day'].values[1], '%Y.%m.%d'),
            'end_date': datetime.datetime.strptime(samples[0]['Day'].values[len(samples[0]['Day'].values) - 1],
                                                   '%Y.%m.%d'),
            'hurst': hurst,
            'cadf': cadf
        }

        print(report)
        ## Store it.
        client = MongoClient('localhost', 27017)
        results = client['series_analysis']
        cresults = results.cointegration
        post_id = cresults.insert_one(report).inserted_id

        return report

    def study_series(self, file, header=['Day', 'Time', 'Open', 'High', 'Low', 'Close', 'Volume']):
        """."""

        ## Prepare data.
        dt = DataPrep()

        df = dt.load_csv(file, header=header)[0]
        file = file.split('/')[-1]
        print('File: ', file)
        print(df.head())
        # df = dt.prepare([df])[0]
        df = dt.merge_daytime(df)
        df = df[(df.index > self.start_date) & (df.index < self.end_date)]

        ## Study data.

        h_exp = hurst(df['Close'].values)
        c_value = adf(df['Close'].values)

        report = {
            'file': file,
            'start_date': datetime.datetime.strptime(df['Day'].values[1], '%Y.%m.%d'),
            'end_date': datetime.datetime.strptime(df['Day'].values[len(df['Day'].values) - 1], '%Y.%m.%d'),
            'hurst': h_exp,
            'adf': c_value
        }

        ## Store it.
        client = MongoClient('localhost', 27017)
        results = client['series_analysis']
        sresults = results.stationarity
        post_id = sresults.insert_one(report).inserted_id
        print(report)

        return report

    def get_files(self, path):
        """."""

        file_paths = []

        for root, dirs, files in os.walk(path):
            if files:
                for file in files:
                    if file.endswith('.csv'):
                        file_path = os.path.join(root, file)
                        print(file_path)
                        file_paths.append(file_path)

        return file_paths

    def study_stationarity(self, files, timeframe = '5.'):
        """."""

        errors = []
        for file in files:
            if timeframe in file:
                try:
                    print(a.study_series(file))
                except Exception as e:
                    errors.append((e, file))

        print(errors)

    def study_cointegration(self, files, timeframe = '5.'):
        """."""

        errors = []
        for afile in files:
            if timeframe in afile:
                for bfile in files:
                    if timeframe in bfile and afile != bfile:
                        try:
                            print(a.study_pair(afile, bfile))
                        except Exception as e:
                            errors.append((e, afile, bfile))

        print(errors)

errors = []
timeframe = '5.'

a = Analyzer(start_date='2018-01-01', end_date='2018-04-01')
files = a.get_files('/home/mcampos/Documents/code/trader/histdata')
print(a.study_cointegration(files))
