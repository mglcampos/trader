import datetime
import json
import os
# import pywt
import pprint
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
import statsmodels.tsa.stattools as ts
from pandas.stats.api import ols
from pykalman import KalmanFilter
# from random import randint
from pymongo import MongoClient
from statsmodels.tsa.seasonal import seasonal_decompose
from talib.abstract import *

from htr.helpers.cointegration import hurst


### momentum = data[t]/data[t-n] - 1 * 100
### simple moving avg = ( price[t] / price[t-n:t].mean() ) - 1 * 100
### bollinger bands = ( price[t] - SMAVG[t] ) / (2*std[t] )   -- if >1 above deviation(sell), if <-1 below deviation(buy)
### sudo /bin/systemctl start grafana-server
def get_dataframe(symbol, filename=None, index=True, mergedIndex=False, removeInter = False, s_file=None):

    ## Get Dataframe indexed by Day. No iterations on the Dataframe.
    t0 = time.clock()
    t1 = time.time()

    s = symbol.replace("/", "")
    # TODO granularity is hardcoded
    if s_file and filename == None:
        symbol_data = pd.io.parsers.read_csv(
            os.path.join('/home/user/Desktop/algotrader/M1/' + s + '/', '% s.txt' % s_file),
            header=0, parse_dates=True,
            names=['Type', 'Day', 'Time', 'Open', 'High', 'Low', 'Close']
        )
    else:
        symbol_data = pd.io.parsers.read_csv(
            os.path.abspath('histdata/' + filename),
            header=0, parse_dates=True,
            names=['Type', 'Day', 'Time', 'Open', 'High', 'Low', 'Close']
        )
    if index == True:
        symbol_data = symbol_data.set_index(symbol_data['Day'].values)
        if mergedIndex == True:
            symbol_data = merge_day_time(symbol_data)

    if removeInter == True:
        symbol_data = remove_interpolations(symbol_data)

    print "# getDataframe # -", time.clock() - t0, "seconds process time"
    print "# getDataframe # -", time.time() - t1, "seconds wall time"

    return symbol_data

def merge_day_time(df):
    t0 = time.clock()
    t1 = time.time()

    date_index = []

    for ir in df.itertuples():
        date = str(ir[2]) + ' ' + str(ir[3])
        # print date
        date = datetime.datetime.strptime(date, "%Y.%m.%d %H:%M")
        date_index.append(date)

    df = df.set_index([date_index])

    print "# merge_day_time # -", time.clock() - t0, "seconds process time"
    print "# merge_day_time # -", time.time() - t1, "seconds wall time"

    return df

def get_monthly_dataframes(symbol, month=None, dataframe=None):
    # returns dict with months as keys
    t0 = time.clock()
    t1 = time.time()

    # if type(dataframe) is pd.DataFrame():
    #     df = dataframe
    # else:
    #     df = get_dataframe(symbol)

    df = dataframe
    ## todo check length

    if type(month) is list:
        mdf = {}
        for m in month:
            # mdf[m] = pd.DataFrame()
            mdf[m] = df_year_to_month(df, m)[m]
    elif month:
        mdf = df_year_to_month(df, month)

    else:
        mdf = df_year_to_month(df)

    print "# get_monthly_dataframes # -", time.clock() - t0, "seconds process time"
    print "# get_monthly_dataframes # -", time.time() - t1, "seconds wall time"

    return mdf

def getDataframeDates(df):
    # print(df.head())
    # print(df.iloc['Day',len(df['Day'][1]) - 1])
    start_date = datetime.datetime.strptime(df['Day'].values[1], '%Y.%m.%d')
    print 'df start_date', start_date
    end_date = datetime.datetime.strptime(df['Day'].values[len(df['Day'].values) - 1], '%Y.%m.%d')
    print 'df end_date', end_date

    return start_date, end_date

def getKalmanFilter(df):
    #requires merged day and time
    t0 = time.clock()
    t1 = time.time()
    series = df['Close']

    kf = KalmanFilter(transition_matrices=[1],
                      observation_matrices=[1],
                      initial_state_mean=0,
                      initial_state_covariance=1,
                      observation_covariance=1,
                      transition_covariance=.01)

    # Use the observed values of the price to get a rolling mean
    state_means, _ = kf.filter(series.values)
    state_means = pd.Series(state_means.flatten(), index=series.index)

    print "# get_kalman_filter # -", time.clock() - t0, "seconds process time"
    print "# get_kalman_filter # -", time.time() - t1, "seconds wall time"

    return state_means

def plot_kalman_filter(df):
    # requires merged day and time
    t0 = time.clock()
    t1 = time.time()

    series = df['Close']
    kf = KalmanFilter(transition_matrices=[1],
                      observation_matrices=[1],
                      initial_state_mean=0,
                      initial_state_covariance=1,
                      observation_covariance=1,
                      transition_covariance=.01)

    # Use the observed values of the price to get a rolling mean
    state_means, _ = kf.filter(series.values)
    state_means = pd.Series(state_means.flatten(), index=series.index)
    # print 'state_means', state_means
    # Compute the rolling mean with various lookback windows
    mean30 = pd.rolling_mean(series, 30)
    mean60 = pd.rolling_mean(series, 60)
    mean90 = pd.rolling_mean(series, 90)

    # Plot original data and estimated mean
    plt.plot(state_means)
    plt.plot(series)
    plt.plot(mean30)
    plt.plot(mean60)
    plt.plot(mean90)
    plt.title('Kalman filter estimate of average')
    plt.legend(['Kalman Estimate', 'X', '30-day Moving Average', '60-day Moving Average', '90-day Moving Average'])
    plt.xlabel('Day')
    plt.ylabel('Price')

    print "# plot_kalman_filter # -", time.clock() - t0, "seconds process time"
    print "# plot_kalman_filter # -", time.time() - t1, "seconds wall time"

    plt.show()

def append_momentum(df):

    df['Momentum'] = float('NaN')
    for i in range(0, df.shape[0]):
        if i >= 5:
            df.loc[i, 'Momentum'] = (df.loc[i, 'Close'] / df.loc[i - 5, 'Close'] - 1) * 100
    return df


def append_rsi(df):
    df = df.reset_index(drop=True)
    delta = df['Close'].diff()
    delta = delta[1:]
    up, down = delta.copy(), delta.copy()
    up[up < 0] = 0
    down[down > 0] = 0

    # Calculate the EWMA
    roll_up1 = pd.stats.moments.ewma(up, 20)
    roll_down1 = pd.stats.moments.ewma(down.abs(), 20)

    # Calculate the RSI based on EWMA
    RS1 = roll_up1 / roll_down1
    df['ERSI20'] = 100.0 - (100.0 / (1.0 + RS1))

    # Calculate the SMA
    roll_up2 = pd.rolling_mean(up, 20)
    roll_down2 = pd.rolling_mean(down.abs(), 20)

    # Calculate the RSI based on SMA
    RS2 = roll_up2 / roll_down2
    df['SRSI20'] = 100.0 - (100.0 / (1.0 + RS2))

    return df

def append_macd(df):
    df['EMA26'] = pd.stats.moments.ewma(df['Close'], 26)
    df['EMA12'] = pd.stats.moments.ewma(df['Close'], 12)
    for ir in df.itertuples():
        df.loc[ir[0], 'MACD'] = (df.loc[ir[0], 'EMA12'] - df.loc[ir[0], 'EMA26']) * 100

    return df

def append_ema(df):

    df['EMA20'] = pd.stats.moments.ewma(df['Close'], 20)

    return df

def append_bollinger(df, window=20):
    w = window

    df['Bollinger'] = float('NaN')
    df['STD'] = pd.rolling_std(df['Close'], window=w)
    rm = pd.rolling_mean(df['Close'], window=w)

    for x in range(0, df.shape[0]):
        df.loc[x, 'Bollinger'] = ((df.loc[x, 'Close'] - rm[x]) / (2 * df.loc[x, 'STD']))
    return df

def append_indicators(df):
    t2 = time.clock()
    t3 = time.time()

    df = df.reset_index(drop=True)

    # rm.plot(label='Rolling Mean', ax=ax)

    rm20 = pd.rolling_mean(df['Close'], window=20)
    df['STD20'] = pd.rolling_std(df['Close'], window=20)
    rm200 = pd.rolling_mean(df['Close'], window=200)
    df['STD200'] = pd.rolling_std(df['Close'], window=200)
    rm50 = pd.rolling_mean(df['Close'], window=50)
    df['STD50'] = pd.rolling_std(df['Close'], window=50)
    rm100 = pd.rolling_mean(df['Close'], window=100)
    df['STD100'] = pd.rolling_std(df['Close'], window=100)
    # rm.plot(label='Rolling Mean', ax=ax)
    ema20 = pd.stats.moments.ewma(df['Close'], 20)
    df['EMA26'] = pd.stats.moments.ewma(df['Close'], 26)
    df['EMA12'] = pd.stats.moments.ewma(df['Close'], 12)

    for ir in df.itertuples():
        # print(datetime.datetime(df.loc[i,'Day'].replace('.','-')))
        # df.loc[i,'Day'] = datetime.datetime(df.loc[i,'Day'].replace('.','-'))
        if ir[0] >= 5:
            df.loc[ir[0], 'Momentum'] = (df.loc[ir[0], 'Close'] / df.loc[ir[0] - 5, 'Close'] - 1) * 100
        df.loc[ir[0], 'SMA20'] = ((df.loc[ir[0], 'Close'] / rm20[ir[0]]) - 1) * 100
        df.loc[ir[0], 'SMA50'] = ((df.loc[ir[0], 'Close'] / rm50[ir[0]]) - 1) * 100
        df.loc[ir[0], 'SMA100'] = ((df.loc[ir[0], 'Close'] / rm100[ir[0]]) - 1) * 100
        df.loc[ir[0], 'SMA200'] = ((df.loc[ir[0], 'Close'] / rm200[ir[0]]) - 1) * 100
        df.loc[ir[0], 'MACD'] = (df.loc[ir[0], 'EMA12'] - df.loc[ir[0], 'EMA26']) * 100
        # df.loc[ir[0], 'Bollinger'] = ((df.loc[ir[0], 'Close'] - rm20[ir[0]]) / (2 * df.loc[ir[0], 'STD20']))
        df.loc[ir[0], 'EMA20'] = ((df.loc[ir[0], 'Close'] / ema20[ir[0]]) - 1) * 100

    print("# append_indicators # -", time.clock() - t2, "seconds process time")
    print("# append_indicators # -", time.time() - t3, "seconds wall time")

    return df


def add_features(df):

    t0 = time.clock()
    t1 = time.time()

    inputs = {
        'open': df['Open'],
        'high': df['High'],
        'low': df['Low'],
        'close': df['Close']
    }

    for t in [3,4,5,8,9,10]:
        df["Momentum%s" % str(t)] = MOM(inputs, timeperiod = t)
    for t in [3, 11, 20, 31, 50, 100, 200]:
        df["SMA%s" % str(t)] = SMA(inputs, timeperiod = t) #add 3,11,31
    for t in [9, 15, 30]:
        macd, macdsignal, macdhist = MACD(inputs, fastperiod=12, slowperiod=26, signalperiod=t) #15, 30
        df["MACD%s" % str(t)] = macd
    for t in [3, 11, 20, 31, 50, 100, 200]:
        df["EMA%s" % str(t)] = EMA(inputs, timeperiod=t)
    for t in [6, 7, 8, 9, 10]:
        df["WILLIAMS%s" % str(t)] = WILLR(inputs, timeperiod=t)
    for t in [12, 13, 14, 15]:
        df["ROCP%s" % str(t)] = ROCP(inputs, timeperiod=t)
    for t in [3, 5, 8, 11, 13, 14, 17, 20, 31]:
        df["ATR%s" % str(t)] = ATR(inputs, timeperiod=t)
        df["RSI%s" % str(t)] = RSI(inputs, timeperiod=t)
        df["ADX%s" % str(t)] = ADX(inputs, timeperiod=t)

    slowk, slowd = STOCH(inputs, fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
    df['STOCHWK'] = slowk
    df['STOCHWD'] = slowd
    df['SIN'] = SIN(inputs)
    df['CCI'] = CCI(inputs, timeperiod=14)
    print("# add_features # -", time.clock() - t0, "seconds process time")
    print("# add_features # -", time.time() - t1, "seconds wall time")

    return df

def df_year_to_month(ydf, month=None):

    t0 = time.clock()
    t1 = time.time()

    start_date, end_date = getDataframeDates(ydf)
    mdf = {}


    if not month:

        for m in range(start_date.month, end_date.month+1):

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

            print 'month_start', month_start
            print 'month_end', month_end

            mdf[str(m)] = pd.DataFrame()
            # mdf[str(m)] = mdf[str(m)].append(ydf.loc[month_start:month_end])
            ydf = ydf.set_index(ydf['Day'].values)
            mdf[str(m)] = ydf.loc[month_start:month_end].drop(month_end)
            mdf[str(m)] = merge_day_time(mdf[str(m)])

        print "# df_year_to_month # -", time.clock() - t0, "seconds process time"
        print "# df_year_to_month # -", time.time() - t1, "seconds wall time"
        return mdf

    else:

        mdf[month] = pd.DataFrame()
        if int(float(month)) == 1:
            if start_date.day < 10:
                month_start = str(start_date.year) + '.0' + str(month) + '.0' + str(start_date.day)
            else:
                month_start = str(start_date.year) + '.0' + str(month) + '.' + str(start_date.day)
            month_end = str(start_date.year) + '.0' + str(int(float(month)) + 1) + '.0' + str(01)

        elif int(float(month)) < 9:
            month_start = str(start_date.year) + '.0' + str(month) + '.0' + str(1)
            month_end = str(start_date.year) + '.0' + str(int(float(month)) + 1) + '.0' + str(1)

        elif int(float(month)) == 9:
            month_start = str(start_date.year) + '.0' + str(month) + '.0' + str(1)
            month_end = str(start_date.year) + '.' + str(int(float(month)) + 1) + '.0' + str(1)

        elif int(float(month)) == 12:
            month_start = str(start_date.year) + '.' + str(month) + '.0' + str(1)
            if end_date.day > 10:
                month_end = str(start_date.year) + '.' + str(month) + '.' + str(end_date.day)
            else:
                month_end = str(start_date.year) + '.' + str(month) + '.0' + str(end_date.day)

        else:
            month_start = str(start_date.year) + '.' + str(month) + '.0' + str(1)
            month_end = str(start_date.year) + '.' + str(int(float(month)) + 1) + '.0' + str(1)

        ydf = ydf.set_index(ydf['Day'].values)
        # print ydf.index.values
        mdf[month] = ydf.loc[month_start:month_end].drop(month_end)
        mdf[str(month)] = merge_day_time(mdf[str(month)])

        # print 'month_start', month_start
        # print 'month_end',  month_end
        # print mdf[month]
        print "# df_year_to_month # -", time.clock() - t0, "seconds process time"
        print "# df_year_to_month # -", time.time() - t1, "seconds wall time"
        return mdf

def save_dataframe(symbol, df, month=None, type=None, granularity='M1'):
    t0 = time.clock()
    t1 = time.time()

    s = symbol.replace("/", "")
    start, end = getDataframeDates(df)

    ##TODO save with granularity

    if month:
        s = s + "_m" + str(month) + "_" + str(end.year)
        df.to_csv(s, header=False, index=False)
    else:
        if not type == None:
            if type == 'Train':
                s = s + '_'+ str(granularity) + "_Train_" + str(end.year)
                df.to_csv(s, header=False, index=False)
            if type == 'Test':
                s = s + '_' + str(granularity) + "_Test_" + str(end.year)
                df.to_csv(s, header=False, index=False)

            if type == 'Concat':
                s = s + '_' + str(granularity) + '_' + str(start.year) + '_' + str(end.year)
                df.to_csv(s, header=False, index=False)
        else:
            s = s + '_' + str(granularity) + "_" + str(end.year)
            df.to_csv(s, header=False, index=False)



    print "# save dataframe # -", time.clock() - t0, "seconds process time"
    print "# save dataframe # -", time.time() - t1, "seconds wall time"

def generate_yearly_out_of_sample(symbol, df, crossval=False, mode='start'):
    testdf = pd.DataFrame()
    traindf = pd.DataFrame()
    ##return 7 months for training and 5 months to testing
    if crossval == False:
        if mode=='start':
            for m in range(1,13):
                if m < 8:
                    traindf = traindf.append(get_monthly_dataframes(symbol, month = str(m), dataframe = df)[str(m)])
                if m >= 8:
                    testdf = testdf.append(get_monthly_dataframes(symbol, month = str(m), dataframe=df)[str(m)])
            return traindf, testdf
        else:
            for m in range(1,13):
                if m < 6:
                    testdf = testdf.append(get_monthly_dataframes(symbol, month=str(m), dataframe=df)[str(m)])
                if m >= 6:
                    traindf = traindf.append(get_monthly_dataframes(symbol, month = str(m), dataframe=df)[str(m)])

            return traindf, testdf
    else:
        pass
        # traindfs = []
        # testdfs = []
        #
        # months = list(range(1,13))
        # move_amount = 1
        # slice_size = 7
        # sets = [months[i:i + slice_size] for i in range(0, 6)]
        # monthly = get_monthly_dataframes(symbol)
        # # print sets
        # for k in sets:
        #     for m in months:
        #         # print k
        #         if m in k:
        #             # print monthly[str(m)]
        #             traindf = traindf.append(monthly[str(m)])
        #         else:
        #             testdf = testdf.append(monthly[str(m)])
        #
        #     traindfs.append(traindf)
        #     testdfs.append(testdf)
        #     testdf = pd.DataFrame()
        #     traindf = pd.DataFrame()
        #
        # return traindfs, testdfs


def remove_interpolations(df, mergedIndex = False):
    t0 = time.clock()
    t1 = time.time()

    df = df.reset_index(drop=True)

    temp = {'Close' : 0, 'Low' : 0, 'High' : 0, 'Open' : 0}
    for ir in df.itertuples():
        if ir[1] == 'I':
            df.loc[ir[0], 'Close'] = temp['Close']
            df.loc[ir[0], 'Low'] = temp['Low']
            df.loc[ir[0], 'High'] = temp['High']
            df.loc[ir[0], 'Open'] = temp['Open']

        else:
            temp = {'Close' : ir[7], 'Low' : ir[6], 'High' : ir[5], 'Open' : ir[4]}

    if mergedIndex == True:
        df = merge_day_time(df)

    print "# remove_interpolations # -", time.clock() - t0, "seconds process time"
    print "# remove_interpolations # -", time.time() - t1, "seconds wall time"


    return df
def merge_yearly_dataframes(dfs):
    cdf = pd.DataFrame()
    for df in dfs:
        cdf = cdf.append(df)

    return cdf

def divide_by_factor(df, factor):

    dfs = []
    start = 0
    df = df.reset_index(drop=True)
    size = df.shape[0] / factor
    while size < df.shape[0]:
        dfs = dfs.append(df.loc[start:size].drop(size))
        size = size + factor
        start = start + factor

    return dfs

def get_m15(df):
    i = 0
    # rdf = pd.DataFrame({'Day' : [], 'Time' : [], 'Open' : [], 'Close' : [], 'Max' : [], 'Min' : [], 'Close' : []})
    rdf = pd.DataFrame()
    df_day = pd.DataFrame()
    df_time = pd.DataFrame()
    df_min = pd.DataFrame()
    df_max = pd.DataFrame()
    df_open = pd.DataFrame()
    df_close = pd.DataFrame()
    df_type = pd.DataFrame()

    min_value = []
    max_value = []
    for ir in df.itertuples():
        if i < 15:
            max_value.append(ir[5])
            min_value.append(ir[6])
            if i == 0:
                df_day = df_day.append(pd.Series(ir[2]), ignore_index=True)
                df_time = df_time.append(pd.Series(ir[3]), ignore_index=True)
                df_open = df_open.append(pd.Series(ir[4]), ignore_index=True)
            elif i == 14:
                df_close = df_close.append(pd.Series(ir[7]), ignore_index=True)
                df_type = df_type.append(pd.Series(ir[1]), ignore_index=True)
                df_max = df_max.append(pd.Series(max(max_value)), ignore_index=True)
                df_min = df_min.append(pd.Series(min(min_value)), ignore_index=True)
                i = -1
                min_value = []
                max_value = []
        else:
            raise ValueError

        i = i + 1


    rdf = pd.concat([df_type, df_day, df_time, df_open, df_max, df_min, df_close], ignore_index=True, axis=1)
    rdf.columns = ['Type','Day', 'Time', 'Open', 'Max', 'Min', 'Close']

    return rdf[:-1]

def get_h1(df):
    i = 0
    # rdf = pd.DataFrame({'Day' : [], 'Time' : [], 'Open' : [], 'Close' : [], 'Max' : [], 'Min' : [], 'Close' : []})
    rdf = pd.DataFrame()
    df_day = pd.DataFrame()
    df_time = pd.DataFrame()
    df_min = pd.DataFrame()
    df_max = pd.DataFrame()
    df_open = pd.DataFrame()
    df_close = pd.DataFrame()
    df_type = pd.DataFrame()

    min_value = []
    max_value = []

    for ir in df.itertuples():
        if i < 60:
            max_value.append(ir[5])
            min_value.append(ir[6])
            if i == 0:
                df_day = df_day.append(pd.Series(ir[2]), ignore_index=True)
                df_time = df_time.append(pd.Series(ir[3]), ignore_index=True)
                df_open = df_open.append(pd.Series(ir[4]), ignore_index=True)
            elif i == 59:
                df_close = df_close.append(pd.Series(ir[7]), ignore_index=True)
                df_type = df_type.append(pd.Series(ir[1]), ignore_index=True)
                df_max = df_max.append(pd.Series(max(max_value)), ignore_index=True)
                df_min = df_min.append(pd.Series(min(min_value)), ignore_index=True)
                i = -1
                min_value = []
                max_value = []
        else:
            raise ValueError

        i = i + 1

    rdf = pd.concat([df_type, df_day, df_time, df_open, df_max, df_min, df_close], ignore_index=True, axis=1)
    rdf.columns = ['Type','Day', 'Time', 'Open', 'Max', 'Min', 'Close']

    return rdf

def get_d1(df):
    i = 0
    # rdf = pd.DataFrame({'Day' : [], 'Time' : [], 'Open' : [], 'Close' : [], 'Max' : [], 'Min' : [], 'Close' : []})
    rdf = pd.DataFrame()
    df_day = pd.DataFrame()
    df_time = pd.DataFrame()
    df_min = pd.DataFrame()
    df_max = pd.DataFrame()
    df_open = pd.DataFrame()
    df_close = pd.DataFrame()
    df_type = pd.DataFrame()

    min_value = []
    max_value = []

    for ir in df.itertuples():
        if i < 1440:
            max_value.append(ir[5])
            min_value.append(ir[6])
            if i == 0:
                df_day = df_day.append(pd.Series(ir[2]), ignore_index=True)
                df_time = df_time.append(pd.Series(ir[3]), ignore_index=True)
                df_open = df_open.append(pd.Series(ir[4]), ignore_index=True)
            elif i == 1439:
                # print ir
                df_close = df_close.append(pd.Series(ir[7]), ignore_index=True)
                df_type = df_type.append(pd.Series(ir[1]), ignore_index=True)
                df_max = df_max.append(pd.Series(max(max_value)), ignore_index=True)
                df_min = df_min.append(pd.Series(min(min_value)), ignore_index=True)
                i = -1
                min_value = []
                max_value = []
        else:
            raise ValueError

        i = i + 1

    rdf = pd.concat([df_type, df_day, df_time, df_open, df_max, df_min, df_close], ignore_index=True, axis=1)
    rdf.columns = ['Type', 'Day', 'Time', 'Open', 'Max', 'Min', 'Close']

    return rdf[:-1]

def generate_sample(symbol):
    symbol = symbol.replace("/", "")
    # years = ['2012','2013','2014','2015', '2016']
    # years = ['2012', '2013']
    years = ['2014', '2015', '2016']

    for y in years:
        filename = 'UDAT_MT_' + str(symbol) + '_M1_'+ str(y)+'.txt'
        df = get_dataframe(symbol, filename=filename)
        df = remove_interpolations(df, mergedIndex=True)

        t0 = time.clock()
        t1 = time.time()

        mdf = get_monthly_dataframes(symbol, dataframe=df)
        for m in mdf.keys():
            save_dataframe(symbol, mdf[m], month=m)
        print "# all get_monthly_dataframes # -", time.clock() - t0, "seconds process time"
        print "# all get_monthly_dataframes # -", time.time() - t1, "seconds wall time"
        t0 = time.clock()
        t1 = time.time()

        dfm15 = get_m15(df)
        save_dataframe(symbol, dfm15, granularity='M15')

        print "# get_m15 # -", time.clock() - t0, "seconds process time"
        print "# get_m15 # -", time.time() - t1, "seconds wall time"
        t0 = time.clock()
        t1 = time.time()

        dfh1 = get_h1(df)
        save_dataframe(symbol, dfh1, granularity='H1')

        print "# get_h1 # -", time.clock() - t0, "seconds process time"
        print "# get_h1 # -", time.time() - t1, "seconds wall time"
        t0 = time.clock()
        t1 = time.time()

        dfd1 = get_d1(df)
        save_dataframe(symbol, dfd1, granularity='D1')

    print "# get_d1 # -", time.clock() - t0, "seconds process time"
    print "# get_d1 # -", time.time() - t1, "seconds wall time"


def decompose_series(df):
    # df = merge_day_time(df)
    ##todo fix this
    # print(np.isnan(np.sum(df['Close'].values)))
    print(df['Close'].head())
    decomposition = seasonal_decompose(df['Close'], freq=1)
    trend = decomposition.trend
    seasonal = decomposition.seasonal
    residual = decomposition.resid


    plt.subplot(411)
    plt.plot(df['Close'], label='Original')
    plt.legend(loc='best')
    plt.subplot(412)
    plt.plot(trend, label='Trend')
    plt.legend(loc='best')
    plt.subplot(413)
    plt.plot(seasonal, label='Seasonality')
    plt.legend(loc='best')
    plt.subplot(414)
    plt.plot(residual, label='Residuals')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()
    return df

def backtest_performance():
    i = 0
    mem = []
    system = []
    user = []
    p = {}
    while i < 100:
        i = i + 1
        mem.append(psutil.virtual_memory()[2])
        system.append(psutil.cpu_times()[2])
        user.append(psutil.cpu_times()[0])
        print('MEMORY ## - ## ' + str(psutil.virtual_memory()) + '\n')
        print('CPU TIMES ## - ## ' + str(psutil.cpu_times()) + '\n')
        print('CPU PERCENT ## - ## ' + str(psutil.cpu_percent()) + '\n')
        print('######################################################\n')
        time.sleep(1)
    p['system'] = system
    p['user'] = user
    p['mem'] = mem
    with open('forecasting_performance.json', 'w') as outfile:
        json.dump(p, outfile)

def evaluate_stationarity(df, df2):
    diff = df['Close'] - df['Close'].shift()
    log = np.log(df['Close'])
    ema = pd.ewma(df['Close'], halflife=12)
    ema_diff = df['Close'] - ema
    res = ols(y=df['Close'], x=df2["Close"])
    beta_hr = res.beta.x
    res = df['Close'] - beta_hr * df2["Close"]
    res2 = ols(y=df2['Close'], x=df["Close"])
    beta_hr = res2.beta.x
    res2 = df2['Close'] - beta_hr * df["Close"]
    kalman = getKalmanFilter(df)
    kalman_diff = kalman - df['Close']
    kalman_hedge = kalman - df2['Close']
    kalman2 = getKalmanFilter(df2)
    kalman_diff2 = kalman2 - df2['Close']
    kalman_hedge2 = kalman2 - df['Close']

    series = [(diff, 'diff'), (log, 'log'), (ema_diff, 'ema_diff'), (res, 'residuals'), (res2, 'residuals2'), (kalman_diff, 'kalman_diff'), (kalman_diff2, 'kalman_diff2'), (kalman_hedge, 'kalman_hedge'), (kalman_hedge2, 'kalman_hedge2')]
    for s in series:
        s = list(s)
        s[0] = s[0][~np.isnan(s[0])]
        print('HURST: '+s[1], hurst(s[0]))
        cadf = ts.adfuller(s[0])
        print(' ADF: '+s[1])
        pprint.pprint(cadf)

def query_mongo():
    client = MongoClient('localhost', 27017)
    results = client['results']
    bresults = results.backtest
    sharpe = []
    profit = []
    for result in bresults.find({"strategy":"BollingerBandsStrategy", "granularity":"M1", "instruments": ["EUR/AUD","EUR/CAD"]}):
    # for result in bresults.find({"strategy": "BollingerBandsStrategy", "granularity": "H1", "instruments": ["EUR/AUD"]}):

        sharpe.append(result['sharpe'])
        profit.append(result['profit'])
    print(sharpe)
    print(profit)
    print("sharpe mean", np.mean(sharpe))
    print("profit mean", np.mean(profit))

query_mongo()

# def denoise(df):
#     signal = pywt.waverec(denoised, 'db8', mode='per')
#
#     fig, axes = plt.subplots(1, 2, sharey=True, sharex=True,
#                              figsize=(10, 8))
#     ax1, ax2 = axes
#
#     ax1.plot(signal)
#     ax1.set_xlim(0, 2 ** 10)
#     ax1.set_title("Recovered Signal")
#     ax1.margins(.1)
#
#     ax2.plot(nblck)
#     ax2.set_title("Noisy Signal")
#
#     for ax in fig.axes:
#         ax.tick_params(labelbottom=False, top=False, bottom=False, left=False,
#                        right=False)
#
#     fig.tight_layout()



##TODO usar csv na pasta principal para testar sempre


# symbol = 'EUR/GBP'
# generate_sample(symbol)

# symbol = 'EUR/USD'
# df1 = get_dataframe(symbol, s_file='UDAT_MT_EURUSD_M1_2015', removeInter=True, index=True)
# df2 = get_dataframe(symbol, s_file='UDAT_MT_EURUSD_M1_2016', removeInter=True, index=True)
#
# print df1.head()
# df = merge_yearly_dataframes([df1, df2])
# print df.head()
# save_dataframe(symbol, df, type='Concat')

# symbol = 'EUR/NZD'
# df2 = get_dataframe(symbol, filename='EURNZD_M15_2014', mergedIndex=True, index=True)
# symbol = 'EUR/AUD'
# df = get_dataframe(symbol, filename='EURAUD_M15_2014', mergedIndex=True, index=True)
# evaluate_stationarity(df, df2)

# df = decompose_series(df[:-1])
# print(df['Close'].head())
# df = add_features(df)
# print df[-250:].head()


# df = append_ema(df)
# df = append_macd(df)
# df = append_rsi(df)
#
# print df[200:].head()



# df = remove_interpolations(df, mergedIndex = True)
# df = get_m15(df)
# print df
# save_dataframe(symbol, df, granularity='M15')

# traindf, testdf = generate_yearly_out_of_sample(symbol, df)
# save_dataframe(symbol, traindf, type='Train')
# save_dataframe(symbol, testdf, type='Test')
# print get_d1(df)


# ydf = get_dataframe(symbol)
# traindfs, testdfs = generate_yearly_out_of_sample(symbol, ydf, crossval=True)
# print testdfs[0]
# print testdfs[2]
# print testdfs[4]
# print testdfs[5]
# print len(traindfs)
# print traindfs
# save_dataframe(symbol, traindf, type='Train')
# save_dataframe(symbol, testdf, type='Test')

# mdf = get_monthly_dataframes(symbol)
# for m in mdf.keys():
#     save_dataframe(symbol, mdf[m], month=m)
# for r in range(1,13):
#     print mdf[str(r)]
# print mdf['12']
# print mdf['3']
# print mdf['9']
# print mdf['10']

#plot_kalman_filter(mdf['1'])
# mdf['1'] = merge_day_time(mdf['1'])
# print getKalmanFilter(mdf['1'])



#

# < media - desceu entao sell, subiu entao buy
# > media - subiu entao buy, desceu entao sell
# l = [1,2,3,4,5,6]
# print l[2:-1]

# backtest_performance()


