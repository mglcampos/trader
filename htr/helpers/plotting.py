
import matplotlib.pyplot as plt
import pandas as pd
import datetime
import numpy as np
import talib

from htr.helpers.dataprep import DataPrep


def moving_average(x, n, type='simple'):
    """
    compute an n period moving average.

    type is 'simple' | 'exponential'

    """
    x = np.asarray(x)
    if type == 'simple':
        weights = np.ones(n)
    else:
        weights = np.exp(np.linspace(-1., 0., n))

    weights /= weights.sum()

    a = np.convolve(x, weights, mode='full')[:len(x)]
    a[:n] = a[n]
    return a


def relative_strength(prices, n=14):
    """
    compute the n period relative strength indicator
    http://stockcharts.com/school/doku.php?id=chart_school:glossary_r#relativestrengthindex
    http://www.investopedia.com/terms/r/rsi.asp
    """

    deltas = np.diff(prices)
    seed = deltas[:n+1]
    up = seed[seed >= 0].sum()/n
    down = -seed[seed < 0].sum()/n
    rs = up/down
    rsi = np.zeros_like(prices)
    rsi[:n] = 100. - 100./(1. + rs)

    for i in range(n, len(prices)):
        delta = deltas[i - 1]  # cause the diff is 1 shorter

        if delta > 0:
            upval = delta
            downval = 0.
        else:
            upval = 0.
            downval = -delta

        up = (up*(n - 1) + upval)/n
        down = (down*(n - 1) + downval)/n

        rs = up/down
        rsi[i] = 100. - 100./(1. + rs)

    return rsi


def moving_average_convergence(x, nslow=26, nfast=12):
    """
    compute the MACD (Moving Average Convergence/Divergence) using a fast and slow exponential moving avg'
    return value is emaslow, emafast, macd which are len(x) arrays
    """
    emaslow = moving_average(x, nslow, type='exponential')
    emafast = moving_average(x, nfast, type='exponential')
    return emaslow, emafast, emafast - emaslow

dt = DataPrep()

file = "/home/mcampos/Documents/code/trader/histdata/crypto/BTCUSD_2017_M10.csv"

df = dt.load_csv(file, header=["Datetime", "High", "Low", "Open", "Close", "Volume", "QuoteVolume", "WeightedAverage"])[0]
df.index = df["Datetime"]
df = df[df.index > '2017-10-26']
df['SAR'] = talib.SAR(df['High'], df['Low'])
emaslow, emafast, df['MACD'] = moving_average_convergence(df['Close'].values)
df['RSI'] = relative_strength(df['Close'].values)
print(df.head())
# Lets plot
fig = plt.figure(1)
fig.suptitle('Trend Following Studies', fontsize=16)
ax = plt.subplot(211)
ax.title.set_text('Price')
df['Close'].plot(legend=None)
ax = plt.subplot(212)
ax.title.set_text('SAR')
df['SAR'].plot(legend=None)

fig = plt.figure(2)
ax = plt.subplot(211)
ax.title.set_text('Price')
df['Close'].plot(legend=None)
fig.subplots_adjust(hspace=1)
ax = plt.subplot(212)
ax.title.set_text('MACD')
df['MACD'].plot(legend=None)

fig = plt.figure(3)
ax = plt.subplot(211)
ax.title.set_text('Price')
df['Close'].plot(legend=None)
fig.subplots_adjust(hspace=1)
ax = plt.subplot(212)
ax.title.set_text('RSI')
df['RSI'].plot(legend=None)
fig = plt.figure(4)
fig.subplots_adjust(hspace=1)
ax = plt.subplot(111)
ax.title.set_text('EMAs')
df['Close'].plot(legend='Price')
pd.Series(emaslow).plot(legend='EMASLOW')
pd.Series(emafast).plot(legend='EMAFAST')
plt.show()



equity=500
for i in range(0,24):
    equity += equity*0.15

    if i < 7:
        equity += 500

    print("Mês: {}, equity: {}".format(i+1, equity))