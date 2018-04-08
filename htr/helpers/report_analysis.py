

from pymongo import MongoClient, ASCENDING

client = MongoClient('localhost', 27017)
backtests = client['backtests']
backtest = backtests.backtest
for test in backtest.find().sort('sharpe', ASCENDING):
    print('Backtest: ', test['name'],  test['files'], ' | Sharpe: ', test['sharpe'], ' | Profitability : ', test['profit'], ' | Max Drawdown: ', test['max_drawdown'])

client = MongoClient('localhost', 27017)
backtests = client['series_analysis']
results = backtests.cointegration
for test in results.find().sort('hurst', ASCENDING):
    print('Files: ', test['files'],  ' | Hurst: ', test['hurst'], ' | CADF : ', test['cadf'])


