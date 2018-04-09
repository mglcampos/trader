# import datetime
# import os
#
# import pandas as pd
# from BrokerHandler import BrokerOanda
# from HistoricalDataHandler import SimpleHistoricDataHandler
# from LiveDataHandler import OandaLiveDataHandler
# from MeanReverting import BollingerBandsStrategy
# from Portfolio import Portfolio
# from Strategies import MovingAverageCrossStrategy
# from histdata.sample.Execution import SimulatedExecutionHandler, OandaExecutionHandler
#
# from htr.core.engines.backtest import Backtest
#
# if __name__ == "__main__":
#
#     ##TODO hearbeat should be 0 if running a simulation
#     ##TODO symbols should always respect the format {0}/{1}, being zero and one currency symbols
#     ##TODO for live strategies, val_type should be bid ou ask, for simulation should be adj_close
#
#     # files = ["_m1_2012","_m1_2013","_m1_2014","_m1_2015","_m1_2016"] + ["_m3_2012","_m3_2013","_m3_2014","_m3_2015","_m3_2016"] + ["_m5_2012","_m5_2013","_m5_2014","_m5_2015","_m5_2016"]
#     files = ["_m1_2012","_m3_2013","_m5_2014","_m1_2015","_m3_2016"]
#     # files = ["_m1_2013","_m3_2012","_m5_2016","_m1_2014","_m3_2014"]
#
#     # files = ['_H1_2016','_H1_2014','_H1_2013','_H1_2012','_H1_2015']
#     # files = ['_M15_2012','_M15_2013','_M15_2014','_M15_2015','_M15_2016']
#     granularity = 'M1'
#
#     for file in files:
#         ###
#         # Set running variables
#         ###
#         # list = ["eur/USD", "eur/AUD",'eur/GBP', "eur/CAD"]
#         list = [['eur/AUD','eur/CAD']]
#         for symbol_list in list:
#             # symbol_list = [symbol_list]
#             # filename = 'EURAUD_m1_2015'
#             store = True
#             month_year = file
#             simulation = True
#             broker = 'Oanda'
#             initial_capital = 10000.0
#
#             if simulation == True:
#                 ###
#                 # Simulations
#                 ###
#                 heartbeat = 0.0
#                 # start_date = datetime.datetime(2014, 1, 2)
#                 s = symbol_list[0].replace("/", "")  ##remove '/' from the string
#                 s_file = s + month_year
#                 ##
#                 #  read from csv
#                 ##
#
#
#                 # df = pd.io.parsers.read_csv(os.path.join(csv_dir, '% s.csv' % s_file), usecols=['Day'], parse_dates=True, names=['Day','Time','Open','Max','Min','adj_close','Volume'])
#
#                 ##
#                 # read from txt //todo refactor this
#                 ##
#                 # df = pd.io.parsers.read_csv(os.path.join(csv_dir + s_file + '/', '% s.txt' % s_file), usecols=['Day'], parse_dates=True,
#                 #     names=['Type', 'Day', 'Time', 'Open', 'Max', 'Min', 'adj_close']
#                 # )
#                 # df = pd.io.parsers.read_csv(os.path.join(csv_dir + s + '/', '% s' % s_file), usecols=['Day'],
#                 #                             parse_dates=True,
#                 #                             names=['Type', 'Day', 'Time', 'Open', 'Max', 'Min', 'adj_close']
#                 #                             )
#
#                 df = pd.io.parsers.read_csv(
#                     os.path.abspath('histdata/' + s_file),
#                     header=0, parse_dates=True, usecols = ['Day'],
#                     names=['Type', 'Day', 'Time', 'Open', 'Max', 'Min', 'Close']
#                 )
#
#
#                 start_date = datetime.datetime.strptime(df['Day'][1], '%Y.%m.%d')
#                 print ('start_date', start_date)
#                 end_date = datetime.datetime.strptime(df['Day'][len(df['Day'])-1], '%Y.%m.%d')
#                 print('end_date', end_date)
#
#                 backtest = Backtest(symbol_list, initial_capital, heartbeat, start_date, SimpleHistoricDataHandler,
#                                     SimulatedExecutionHandler, Portfolio, BollingerBandsStrategy, period=granularity, length=len(df), s_file=month_year, store=store)
#
#                 # backtest = Backtest(symbol_list, initial_capital, heartbeat, start_date, HistoricMT4CSVDataHandler,
#                 #                     SimulatedExecutionHandler, Portfolio, BollingerBandsStrategy, csv_dir=csv_dir, period=granularity)
#
#                 # backtest = Backtest(symbol_list, initial_capital, heartbeat, start_date, HistoricMT4CSVDataHandler,
#                 #                     SimulatedExecutionHandler, Portfolio, IntradayOLSMRStrategy, csv_dir=csv_dir,
#                 #                     period=granularity)
#
#
#
#                 # backtest = Backtest(symbol_list, initial_capital, heartbeat, start_date, HistoricMT4CSVDataHandler,
#                 #                     SimulatedExecutionHandler, Portfolio, MovingAverageCrossStrategy, csv_dir=csv_dir, period=granularity)
#
#             elif simulation == False:
#                 ###
#                 # Live Trading
#                 ###
#                 start_date = datetime.datetime.now()
#                 heartbeat = 5
#                 if broker == 'Oanda':
#                     try:
#                         oanda = BrokerOanda()
#                         initial_capital = oanda.get_capital()
#                         backtest = Backtest(symbol_list, initial_capital, heartbeat, start_date, OandaLiveDataHandler,
#                                             OandaExecutionHandler, Portfolio, MovingAverageCrossStrategy, broker=oanda,
#                                              simulation=False)
#                     except Exception as e:
#                         print ("Couldn't connect to Oanda - ", e)
#
#
#             try:
#                 backtest.simulate_trading()
#             except Exception:
#                 print('error in: ', file, symbol_list)