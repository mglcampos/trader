import datetime

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import pprint
import statsmodels.tsa.stattools as ts
from pandas.stats.api import ols

#  Different types of time series for testing
    #p = log10(cumsum(random.randn(50000)+1)+1000) # trending, hurst ~ 1
    #p = log10((random.randn(50000))+1000)   # mean reverting, hurst ~ 0
    # p = log10(cumsum(random.randn(50000))+1000) # random walk, hurst ~ 0.5

import numpy as np

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

def plot_price_series(ts1, ts2, symbol_list, start_date, end_date):
    months = mdates.MonthLocator() # every month
    fig, ax = plt.subplots()
    ax.plot(df.index, ts1, label=symbol_list[0])
    ax.plot(df.index, ts2, label=symbol_list[1])
    # ax.xaxis.set_major_locator(months)
    # ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    # ax.set_xlim(start_date, end_date)
    ax.grid(True)
    # fig.autofmt_xdate()
    plt.xlabel('Month/Year')
    plt.ylabel('Price ($)')
    plt.title('%s and %s Daily Prices' % (symbol_list[0], symbol_list[1]))
    plt.legend()
    plt.show()
    
def plot_scatter_series(ts1, ts2, symbol_list):
    plt.xlabel('%s Price ($)' % symbol_list[0])
    plt.ylabel('%s Price ($)' % symbol_list[1])
    plt.title('%s and %s Price Scatterplot' % (symbol_list[0], symbol_list[1]))
    plt.scatter(ts1, ts2)
    plt.show()
    
def plot_residuals(df, start_date, end_date):
    months = mdates.MonthLocator() # every month
    fig, ax = plt.subplots()
    ax.plot(df.index, df["res"], label="Residuals")
    # ax.xaxis.set_major_locator(months)
    # ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    # ax.set_xlim(start_date, end_date)
    ax.grid(True)
    # fig.autofmt_xdate()
    plt.xlabel('Month/Year')
    plt.ylabel('Price ($)')
    plt.title('Residual Plot')
    plt.legend()
    plt.plot(df["res"])
    plt.show()

if __name__ == "__main__":
    # start = datetime.datetime(2012, 1, 1)
    # end = datetime.datetime(2013, 1, 1)
    # arex = web.DataReader("AREX", "yahoo", start, end)
    # wll = web.DataReader("WLL", "yahoo", start, end)
    # df = pd.DataFrame(index=arex.index)
    # df["AREX"] = arex["Adj Close"]
    # df["WLL"] = wll["Adj Close"]

    symbol_list = ['EURUSD', 'EURAUD']
    s_file = '_H1_2012'
    file1 = symbol_list[0] + s_file
    file2 = symbol_list[1] + s_file
    df1 = pd.read_csv("histdata/" + file1, usecols=['Day', 'Open', 'High', 'Low', 'Close'],
                     names=['Type', 'Day', 'Time', 'Open', 'High', 'Low', 'Close'])
    df2 = pd.read_csv("histdata/" + file2, usecols=['Day', 'Open', 'High', 'Low', 'Close'],
                     names=['Type', 'Day', 'Time', 'Open', 'High', 'Low', 'Close'])
    # print(df1.head())
    # print(df2.head())
    df = pd.DataFrame(index=df1.index)
    start_date = datetime.datetime.strptime(df1['Day'][1], '%Y.%m.%d')
    print('start_date', start_date)
    end_date = datetime.datetime.strptime(df1['Day'][len(df1['Day']) - 1], '%Y.%m.%d')
    print('end_date', end_date)

    # Calculate optimal hedge ratio "beta"
    res = ols(y=df1['Close'], x=df2["Close"])
    print('residuals', res)
    print('beta', res.beta)
    print('beta_hr', res.beta.x)
    beta_hr = res.beta.x
    # Calculate the residuals of the linear combination
    # df["res"] = df["WLL"] - beta_hr*df["AREX"]
    df["res"] = df1['Close'] - beta_hr * df2["Close"]
    hurst = hurst(df['res'][:-1].values)
    print('\n0Hurst Exponent')
    pprint.pprint(hurst)
    # print df
    # Calculate and reports the CADF test on the residuals
    cadf = ts.adfuller(df["res"][:-1])
    pprint.pprint(cadf)
    # Plot the two time series
    plot_price_series(df1['Close'].values, df2["Close"].values, symbol_list, start_date, end_date)
    # Display a scatter plot of the two time series
    plot_scatter_series(df1['Close'].values, df2["Close"].values, symbol_list)
    # Plot the residuals
    plot_residuals(df, start_date, end_date)

