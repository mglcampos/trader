# import datetime
import pprint
# import schedule
import json
import psutil

try:
	import Queue as queue
except ImportError:
	import queue
import time


class Backtest():
	"""
	Event-driven backtest.
	"""

	def __init__(self, data_handler, execution_handler, portfolio, strategy, risk_handler, context, heartbeat = 0):
		"""
		Initialises the backtest.
		Parameters:
		csv_dir - The hard root to the CSV data directory, only necessary for HistoricCSVDataHandler
		symbol_list - The list of symbol strings.
		initial_capital - The starting capital for the portfolio.
		heartbeat - Backtest "heartbeat" in seconds
		start_date - The start datetime of the strategy.
		handler - (Class) Handles the market data feed.
		execution_handler - (Class) Handles the orders/fills for trades.
		portfolio - (Class) Keeps track of portfolio current
		and prior positions.
		strategy - (Class) Generates signals based on market data.
		simulation - true or false
		period - tick granularity
		"""

		self.context = context
		self.time0 = time.clock()
		self.time1 = time.time()
		self.data_handler_cls = data_handler
		self.execution_handler_cls = execution_handler
		self.portfolio_cls = portfolio
		self.strategy_cls = strategy
		self.risk_handler_cls = risk_handler
		self.heartbeat = heartbeat
		self.events = queue.Queue()
		self.signals = 0
		self.orders = 0
		self.fills = 0
		self.num_strats = 1
		# self.granularities = {'S1': 1, 'S5': 5, 'S10':10, 'S15':15, 'S30':30, 'M1':60, 'M2': 120, 'M5': 60*5}
		self._instance_factory()

	def _instance_factory(self):
		"""
		Generates the trading instance objects from their class types.
		"""

		print("Creating Data Handler, Strategy, Portfolio, Risk Handler and Execution Handler")

		self.data_handler = self.data_handler_cls(self.context, self.events)
		self.strategy = self.strategy_cls(self.context, self.events, self.data_handler)
		self.risk_handler = self.risk_handler_cls(self.context)
		self.portfolio = self.portfolio_cls(self.context, self.events, self.risk_handler, self.data_handler)
		self.execution_handler = self.execution_handler_cls(self.context, self.events)
		print('STRATEGY', self.context.strategies)


	def _run_backtest(self):
		"""
		Executes the backtest.
		"""

		while True:
			# Update the market bars
			if self.data_handler.continue_backtest == True:
				self.data_handler.update_bars()
			else:
				break
			# Handle the events
			while True:

				try:
					event = self.events.get(False)
				except queue.Empty:
					break
				else:
					if event is not None:
						print('EVENT', event.type)
						# print event.type
						if event.type == 'MARKET':
							# t2 = time.clock()
							# t3 = time.time()
							# self.cpu.append(psutil.cpu_percent(interval=None))
							# self.cpu_times.append(psutil.cpu_times())
							self.strategy.calculate_signals()
							# print("# market # -", time.clock() - t2, "seconds process time")
							# print("# market # -", time.time() - t3, "seconds wall time")
							# self.process_time.append(time.clock() - t2)
							# self.wall_time.append(time.time() - t3)
							self.portfolio.update_timeindex(event)
							## todo group signal too
						elif event.type == 'SIGNAL' or event.type == 'GROUP_SIGNAL':
							self.signals += 1
							self.portfolio.process_signal(event)
						elif event.type == 'ORDER':
							self.orders += 1
							self.execution_handler.execute_order(event)
						elif event.type == 'FILL':
							self.fills += 1
							self.portfolio.update_fill(event)

			time.sleep(self.heartbeat)


	def _output_performance(self):
		print("#BACKTEST # -", time.clock() - self.time0, "seconds process time")
		print("#BACKTEST # -", time.time() - self.time1, "seconds wall time")

		"""
		Outputs the strategy performance from the backtest.
		"""
		self.portfolio.create_equity_curve_dataframe()
		print("Creating summary stats...")
		stats = self.portfolio.output_summary_stats(self.data_handler.get_symbol_data())
		print("Creating equity curve...")
		print(self.portfolio.equity_curve.tail(10))
		pprint.pprint(stats)
		print("Signals: %s" % self.signals)
		print("Orders: %s" % self.orders)
		print("Fills: %s" % self.fills)
		# plt.show()


	def simulate_trading(self):
		"""
		Simulates the backtest and outputs performance.
		"""
		# t0 = time.clock()
		# t1 = time.time()
		self._run_backtest()
		# cputime = {}
		# cputime['process_time'] = self.process_time
		# cputime['wall_time'] = self.wall_time
		# cputime['cpu'] = self.cpu
		# cputime['cputimes'] = self.cpu_times
		# with open('cpu_time_forecasting.json', 'w') as outfile:
		#     json.dump(cputime, outfile)
		# print "# simulation # -", time.clock() - t0, "seconds process time"
		# print "# simulation # -", time.time() - t1, "seconds wall time"
		self._output_performance()
