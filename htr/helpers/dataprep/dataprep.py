
import pandas as pd
import datetime
import os
import time


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
				dframes[i] = self.merge_daytime(df)
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

	def merge_daytime(self, df):
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
			self.header = ['Day', 'Time', 'Open', 'High', 'Low', 'Close']
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

	def get_monthly_dataframes(self, df, month=None):
		# returns dict with months as keys
		t0 = time.clock()
		t1 = time.time()

		## todo check length

		if type(month) is list:
			mdf = {}
			for m in month:
				# mdf[m] = pd.DataFrame()
				mdf[m] = self.df_year_to_month(df, m)[m]
		elif month:
			mdf = self.df_year_to_month(df, month)

		else:
			mdf = self.df_year_to_month(df)

		print("# get_monthly_dataframes # -", time.clock() - t0, "seconds process time")
		print("# get_monthly_dataframes # -", time.time() - t1, "seconds wall time")

		return mdf

	def get_df_dates(self, df):
		# print(df.head())
		# print(df.iloc['Day',len(df['Day'][1]) - 1])
		start_date = datetime.datetime.strptime(df['Day'].values[1], '%Y.%m.%d')
		print('df start_date', start_date)
		end_date = datetime.datetime.strptime(df['Day'].values[len(df['Day'].values) - 1], '%Y.%m.%d')
		print('df end_date', end_date)

		return start_date, end_date

	def df_year_to_month(self, ydf, month=None):

		t0 = time.clock()
		t1 = time.time()

		start_date, end_date = self.get_df_dates(ydf)
		mdf = {}

		if month is None:

			for m in range(start_date.month, end_date.month + 1):

				if m == 1:
					if start_date.day < 10:
						month_start = str(start_date.year) + '.0' + str(m) + '.0' + str(start_date.day)
					else:
						month_start = str(start_date.year) + '.0' + str(m) + '.' + str(start_date.day)

					month_end = str(start_date.year) + '.0' + str(m + 1) + '.0' + str(1)
				elif m < 9:
					month_start = str(start_date.year) + '.0' + str(m) + '.0' + str(1)
					month_end = str(start_date.year) + '.0' + str(m + 1) + '.0' + str(1)
				elif m == 9:
					month_start = str(start_date.year) + '.0' + str(m) + '.0' + str(1)
					month_end = str(start_date.year) + '.' + str(m + 1) + '.0' + str(1)

				elif m == 12:
					month_start = str(start_date.year) + '.' + str(m) + '.0' + str(1)
					if end_date.day > 10:
						month_end = str(start_date.year) + '.' + str(m) + '.' + str(end_date.day)
					else:
						month_end = str(start_date.year) + '.' + str(m) + '.0' + str(end_date.day)

				else:
					month_start = str(start_date.year) + '.' + str(m) + '.0' + str(1)
					month_end = str(start_date.year) + '.' + str(m + 1) + '.0' + str(1)

				print('month_start', month_start)
				print('month_end', month_end)

				mdf[str(m)] = pd.DataFrame()
				# mdf[str(m)] = mdf[str(m)].append(ydf.loc[month_start:month_end])
				ydf = ydf.set_index(ydf['Day'].values)
				mdf[str(m)] = ydf.loc[month_start:month_end].drop(month_end)
				mdf[str(m)] = self._merge_day_time(mdf[str(m)])

			print("# df_year_to_month # -", time.clock() - t0, "seconds process time")
			print("# df_year_to_month # -", time.time() - t1, "seconds wall time")

			return mdf

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

	def resize(self, df1, df2):
		"""."""

		diff = len(df1) - len(df2)
		if diff > 0:
			df1 = df1[:-diff]
		elif diff < 0:
			df2 = df2[:diff]

		return [df1, df2]

class Evaluation():
	def __init__(self):
		self.merged = None
		self.gaps = None

