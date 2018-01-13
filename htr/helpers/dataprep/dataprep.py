
import pandas as pd
import datetime
import os


class DataPrep():
	def __init__(self, context=None):
		if context is not None:
			self.header = context.data_header
			self.prep_flag = context.data_preparation

	def prepare(self, dframes, fill_mode='ffill'):

		for i, df in enumerate(dframes):

			if type(df.index) == pd.DatetimeIndex:
				try:
					df.index.freqstr
				except:
					freq = (df.index[1] - df.index[0]).seconds
					dframes[i] = self._fill_gaps(df, freq, mode=fill_mode)

			else:
				dframes[i] = self._merge_daytime(df)
				try:
					df.index.freqstr
				except:
					freq = (dframes[i].index[1] - dframes[i].index[0]).seconds
					dframes[i] = self._fill_gaps(dframes[i], freq, mode=fill_mode)

		return dframes

	def _fill_gaps(self, df, freq, mode='ffill'):
		rule = self._get_rule(freq)
		if mode == 'ffill':
			df = df.resample(rule).ffill()
		else:
			pass
		##todo may not want to remove first line
		return df[1:]

	def _add_headers(self, df):
		## TODO get headers from config file
		nrcolumns = len(df.columns)
		pass

	def _merge_daytime(self, df):
		'''

		Args:
			dataframe: pandas dataframe

		Returns:
			Merged dataframe

		'''

		# TODO make sure the right columns are being merged and time has the right format

		self.header = list(map(lambda x: x.upper(), self.header))
		# take index into account+1
		day = self.header.index('day'.upper()) + 1
		time = self.header.index('time'.upper()) + 1

		date_index = []
		for ir in df.itertuples():
			date = str(ir[day]) + ' ' + str(ir[time])
			# print date
			date = datetime.datetime.strptime(date, "%Y.%m.%d %H:%M")
			date_index.append(date)

		df = df.set_index([date_index])

		return df

	def evaluate_data(self, df):
		# check for gaps, merged daytimeindex, header names, spikes
		eval = Evaluation()
		##check for merged index

	##todo assess if makes sense having this
	# df = self._merge_daytime(df)
	#
	# if type(df.index) == pd.DatetimeIndex:
	#     setattr(eval, 'merged', True)
	#     if df.index.freqstr is None:
	#         print((df.index[1] - df.index[0]).seconds)
	# else:
	#     setattr(eval, 'merged', False)
	#
	# print(df.head())
	# print(df.columns)
	#
	# return eval


	def _get_rule(self, freq):
		rule = ''
		if freq == 1:
			rule = 'S'
		elif round(freq / 60) == 1:
			rule = 'T'
		elif round(freq / 60) > 1 and freq < 3600:
			rule = '{}T'.format(str(round(freq / 60)))
		elif round(freq / 3600) == 1:
			rule = 'H'
		elif round(freq / 3600) > 1 and freq < 3600 * 24:
			rule = '{}H'.format(str(round(freq / 3600)))
		elif round(freq / (3600 * 24)) == 1:
			rule = 'D'
		elif round(freq / (3600 * 24)) > 1 and freq < 3600 * 24 * 7:
			rule = '{}D'.format(str(round(freq / (3600 * 24))))

		return rule

	def load_csv(self, path, header=None):
		if header is None:
			self.header = ['Type', 'Day', 'Time', 'Open', 'High', 'Low', 'Close']
		else:
			self.header = header

		# if path is not None:
		#     df = pd.read_csv(path, names=self.header)
		#     # print(len(df.columns))
		#     # print(df.head())
		# else:
		#     pass
		dframes = []
		if path is not None and os.path.isdir(path):
			for root, dirs, files in os.walk(path):
				for file in files:
					if file.endswith('.csv') or file.endswith('.txt'):
						dframes.append(pd.read_csv(os.path.join(root, file), names=self.header))
		elif os.path.isfile(path):
			dframes.append(pd.read_csv(path, names=self.header))

		return dframes

	def load_crypto(self, path, header=None):
		if header is None:
			self.header = ['Type', 'Day', 'Time', 'Open', 'High', 'Low', 'Close']
		else:
			self.header = header

		# if path is not None:
		#     df = pd.read_csv(path, names=self.header)
		#     # print(len(df.columns))
		#     # print(df.head())
		# else:
		#     pass
		dframes = []
		if path is not None and os.path.isdir(path):
			for root, dirs, files in os.walk(path):
				for file in files:
					if file.endswith('.csv') or file.endswith('.txt'):
						dframes.append(pd.read_csv(os.path.join(root, file), index_col='date'))
		elif os.path.isfile(path):
			dframes.append(pd.read_csv(path, index_col='date'))

		return dframes

	def index_union(self, dframes):

		index = None

		for dframe in dframes:
			if index is None:
				index = dframe.index
			else:
				index.union(dframe.index)

		for k, df in enumerate(dframes):
			df[k] = df[k].reindex(index=index, method='pad')

		return dframes


class Evaluation():
	def __init__(self):
		self.merged = None
		self.gaps = None

