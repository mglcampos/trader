
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.tsa.stattools as ts
import pprint

def hurst(p):
    tau = []; lagvec = []
    #  Step through the different lags
    for lag in range(2,20):
        #  produce price difference with lag
        pp = np.subtract(p[lag:],p[:-lag])
        #  Write the different lags into a vector
        lagvec.append(lag)
        #  Calculate the variance of the difference vector
        tau.append(np.sqrt(np.std(pp)))
    #  linear fit to double-log graph (gives power)
    m = np.polyfit(np.log10(lagvec),np.log10(tau),1)
    # calculate hurst
    hurst = m[0]*2
    return hurst

def evaluate_stationarity(df):
    diff = df['close'] - df['close'].shift()
    log = np.log(df['close'])
    ema = pd.ewma(df['close'], halflife=12)
    ema_diff = df['close'] - ema

    # series = [(diff, 'diff'), (log, 'log'), (ema_diff, 'ema_diff'), (df['close'], 'price')]
    series = [(df['close'], 'price')]

    for s in series:
        s = list(s)
        s[0] = s[0][~np.isnan(s[0])]
        print('HURST: '+s[1], hurst(s[0]))
        cadf = ts.adfuller(s[0])
        print(' ADF: '+s[1])
        pprint.pprint(cadf)

def decompose_series(df):
    # df = merge_day_time(df)
    ##todo fix this
    # print(np.isnan(np.sum(df['Close'].values)))
    df.reset_index(inplace=True)
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')

    decomposition = seasonal_decompose(df['close'], freq=1)
    trend = decomposition.trend
    seasonal = decomposition.seasonal
    residual = decomposition.resid

    plt.subplot(411)
    plt.plot(df['close'], label='Original')
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

# from htr.helpers.dataprep import DataPrep
# import datetime
# dt = DataPrep()
# # df = dt.load_crypto("../../histdata/USDT_BTC.csv",
# #                          header=["date, high, low, open, close, volume, quoteVolume, weightedAverage"])[0]
# #
# # df.index = df.index.map(lambda x: datetime.datetime.fromtimestamp(int(x)).strftime('%Y-%m-%d %H:%M:%S'))
# # df = df.dropna()
#
# df = pd.read_csv("../../histdata/XRPUSD_all.csv", names=['date', 'high', 'low', 'open', 'close', 'volume', 'quoteVolume', 'weightedAverage'], index_col = 'date', parse_dates=True)
#
# df = df.dropna()
#
# dfs = []
# dfs.append(df[df.index > '2018-02-01'])
# dfs.append(df[(df.index > '2018-01-01') & (df.index < '2018-02-01')])
# dfs.append(df[(df.index > '2017-12-01') & (df.index < '2018-01-01')])
# dfs.append(df[(df.index > '2017-11-01') & (df.index < '2017-12-01')])
# dfs.append(df[(df.index > '2017-10-01') & (df.index < '2017-11-01')])
# dfs.append(df[(df.index > '2017-09-01') & (df.index < '2017-10-01')])
#
# for frame in dfs:
#     try:
#         print('\n############################################################################################################\n')
#         print(frame.index[0])
#         print(evaluate_stationarity(frame))
#     except:
#         continue
