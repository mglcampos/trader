

from pymongo import MongoClient, ASCENDING

client = MongoClient('localhost', 27017)

# backtests = client['backtests']
# backtest = backtests.backtest
# for test in backtest.find().sort('sharpe', ASCENDING):
#     if 'forex' in test['files'][0]['path'][0]:
#         print('Backtest: ', test['name'],  test['files'], ' | Sharpe: ', test['sharpe'], ' | Profitability : ', test['profit'], ' | Max Drawdown: ', test['max_drawdown'])
#     elif 'crypto_mt' in test['files'][0]['path'][0]:
#         print('Backtest: ', test['name'],  test['files'], ' | Sharpe: ', test['sharpe'], ' | Profitability : ', test['profit'], ' | Max Drawdown: ', test['max_drawdown'])
#
# backtests = client['series_analysis']
# results = backtests.cointegration
# for test in results.find().sort('hurst', ASCENDING):
#     print('Cointegration: ', test['files'],  ' | Hurst: ', test['hurst'], ' | CADF : ', test['cadf'])

backtests = client['series_analysis']
sresults = backtests.stationarity
for test in sresults.find().sort('hurst', ASCENDING):
    print('Stationarity: ', test['file'],  ' | Hurst: ', test['hurst'], ' | ADF : ', test['adf'])


