
from htr.helpers.dataprep import DataPrep
import datetime

dt = DataPrep()
btc = dt.load_crypto("../../../histdata/USDT_BTC.csv",
                         header=["date, high, low, open, close, volume, quoteVolume, weightedAverage"])
# btc[0].index = btc[0].index.map(lambda x: datetime.datetime.fromtimestamp(int(x)).strftime('%Y-%m-%d %H:%M:%S'))
eth = dt.load_crypto("../../../histdata/BTC_ETH.csv",
                     header=["date, high, low, open, close, volume, quoteVolume, weightedAverage"])
ltc = dt.load_crypto("../../../histdata/BTC_LTC.csv",
                     header=["date, high, low, open, close, volume, quoteVolume, weightedAverage"])
xrp = dt.load_crypto("../../../histdata/BTC_XRP.csv",
                     header=["date, high, low, open, close, volume, quoteVolume, weightedAverage"])
etc = dt.load_crypto("../../../histdata/BTC_ETC.csv",
                     header=["date, high, low, open, close, volume, quoteVolume, weightedAverage"])

columns = ['high', 'low', 'open', 'close']
index = btc[0].index
index.union(eth[0].index)
index.union(ltc[0].index)
index.union(xrp[0].index)
index.union(etc[0].index)

eth[0] = eth[0].reindex(index=index, method='pad')
ltc[0] = ltc[0].reindex(index=index, method='pad')
xrp[0] = xrp[0].reindex(index=index, method='pad')
etc[0] = etc[0].reindex(index=index, method='pad')
btc[0] = btc[0].reindex(index=index, method='pad')

eth[0].index = eth[0].index.map(lambda x: datetime.datetime.fromtimestamp(int(x)).strftime('%Y-%m-%d %H:%M:%S'))
btc[0].index = btc[0].index.map(lambda x: datetime.datetime.fromtimestamp(int(x)).strftime('%Y-%m-%d %H:%M:%S'))
xrp[0].index = xrp[0].index.map(lambda x: datetime.datetime.fromtimestamp(int(x)).strftime('%Y-%m-%d %H:%M:%S'))
ltc[0].index = ltc[0].index.map(lambda x: datetime.datetime.fromtimestamp(int(x)).strftime('%Y-%m-%d %H:%M:%S'))
etc[0].index = etc[0].index.map(lambda x: datetime.datetime.fromtimestamp(int(x)).strftime('%Y-%m-%d %H:%M:%S'))

for column in columns:
	# btc[0][column] = btc[0][column].map(lambda x: 1 / x)
	eth[0][column] = eth[0][column] * btc[0][column]
	ltc[0][column] = ltc[0][column] * btc[0][column]
	xrp[0][column] = xrp[0][column] * btc[0][column]
	etc[0][column] = etc[0][column] * btc[0][column]

print(btc[0].head())
print(eth[0].head())
print(ltc[0].head())
print(xrp[0].head())


btc[0] = btc[0].dropna()
eth[0] = eth[0].dropna()
etc[0] = etc[0].dropna()
ltc[0] = ltc[0].dropna()
xrp[0] = xrp[0].dropna()

'''
btc[0][2:].to_csv('/home/mcampos/Documents/code/algotrader/histdata/BTCUSD_all.csv', encoding='utf-8', header=False)
eth[0][2:].to_csv('/home/mcampos/Documents/code/algotrader/histdata/ETHUSD_all.csv', encoding='utf-8', header=False)
xrp[0][2:].to_csv('/home/mcampos/Documents/code/algotrader/histdata/XRPUSD_all.csv', encoding='utf-8', header=False)
ltc[0][2:].to_csv('/home/mcampos/Documents/code/algotrader/histdata/LTCUSD_all.csv', encoding='utf-8', header=False)
etc[0][2:].to_csv('/home/mcampos/Documents/code/algotrader/histdata/ETCUSD_all.csv', encoding='utf-8', header=False)
'''

btc[0][btc[0].index > '2017-11-01'].to_csv('../../../histdata/BTCUSD_2017_M11M12.csv', encoding='utf-8', header=False)
btc[0][btc[0].index > '2017-12-16'].to_csv('../../../histdata/BTCUSD_2017_M12.csv', encoding='utf-8', header=False)
eth[0][eth[0].index > '2017-11-01'].to_csv('../../../histdata/ETHUSD_2017_M11M12.csv', encoding='utf-8', header=False)
eth[0][eth[0].index > '2017-12-16'].to_csv('../../../histdata/ETHUSD_2017_M12.csv', encoding='utf-8', header=False)
xrp[0][xrp[0].index > '2017-11-01'].to_csv('../../../histdata/XRPUSD_2017_M11M12.csv', encoding='utf-8', header=False)
xrp[0][xrp[0].index > '2017-12-16'].to_csv('../../../histdata/XRPUSD_2017_M12.csv', encoding='utf-8', header=False)
ltc[0][ltc[0].index > '2017-11-01'].to_csv('../../../histdata/LTCUSD_2017_M11M12.csv', encoding='utf-8', header=False)
ltc[0][ltc[0].index > '2017-12-16'].to_csv('../../../histdata/LTCUSD_2017_M12.csv', encoding='utf-8', header=False)
etc[0][etc[0].index > '2017-11-01'].to_csv('../../../histdata/ETCUSD_2017_M11M12.csv', encoding='utf-8', header=False)
