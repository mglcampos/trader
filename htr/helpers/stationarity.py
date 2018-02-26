
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
    diff = df['Close'] - df['Close'].shift()
    log = np.log(df['Close'])
    ema = pd.ewma(df['Close'], halflife=12)
    ema_diff = df['Close'] - ema

    series = [(diff, 'diff'), (log, 'log'), (ema_diff, 'ema_diff'), (df['Close'], 'price')]

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


## todo load data
## todo for each df evaluate_stationarity
print('\n############################################################################################################\n')