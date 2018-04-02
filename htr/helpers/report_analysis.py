

from pymongo import MongoClient, ASCENDING

client = MongoClient('localhost', 27017)
backtests = client['backtests']
backtest = backtests.backtest
for test in backtest.find().sort('sharpe', ASCENDING):
    print('Backtest: ', test['name'],  test['files'], ' | Sharpe: ', test['sharpe'], ' | Profitability : ', test['profit'], ' | Max Drawdown: ', test['max_drawdown'])


