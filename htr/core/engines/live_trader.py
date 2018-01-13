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
from LiveDataHandler import OandaLiveDataHandler
from BrokerHandler import BrokerOanda


class LiveTrader:
    """
    Event-driven backtest.
    """

    def __init__(self, data_handler, execution_handler, portfolio, strategy, risk_handler, broker_handler, context):
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

        self.symbol_list = context.symbol_list
        self.heartbeat = context.heartbeat
        self.data_handler_cls = data_handler
        self.execution_handler_cls = execution_handler
        self.portfolio_cls = portfolio
        self.strategy_cls = strategy
        ## todo add in factory
        self.broker_handler_cls = broker_handler
        self.events = queue.Queue()
        self.signals = 0
        self.orders = 0
        self.fills = 0
        self.num_strats = 1


        self._set_up()

    def _set_up(self):
        """
        Generates the trading instance objects from
        their classes and sets up the trading environment.
        """

        self.initial_capital = self.broker_handler.get_initial_capital()
        setattr(self.context, 'initial_capital', self.initial_capital)

        print(
            "Creating DataHandler, Strategy, Portfolio and ExecutionHandler"
        )
        # self.handler = self.data_handler_cls(self.events, self.csv_dir,
        #                                           self.symbol_list)
        # self.data_handler = self.data_handler_cls(self.symbol_list, csv_dir = self.csv_dir,
        #                                           start = self.start_date, events = self.events, s_file=self.s_file)
        #
        # self.strategy = self.strategy_cls(self.data_handler, self.events)
        # self.portfolio = self.portfolio_cls(self.data_handler, self.events,
        #                                     self.start_date,
        #                                     self.initial_capital, simulation=self.simulation, store=self.store, granularity=self.period, strategy=self.strategy.__class__.__name__)
        # self.execution_handler = self.execution_handler_cls(self.events, oanda=self.broker)

        self.data_handler = self.data_handler_cls(self.context, self.events)
        self.strategy = self.strategy_cls(self.context, self.events, self.data_handler)
        self.risk_handler = self.risk_handler_cls()
        self.portfolio = self.portfolio_cls(self.context, self.events, self.risk_handler, self.data_handler)
        self.execution_handler = self.execution_handler_cls(self.context, self.events)
        ## todo finish this
        self.broker_handler = self.broker_handler_cls()

    def _run(self):
        """
        Executes the backtest.
        """

        i = 0
        while True:
            i += 1
            print i
            if isinstance(self.data_handler, OandaLiveDataHandler):
                # Start tick streaming
                if i == 1:
                    print('Starting Tick Listener')
                    self.data_handler.startListener()

                self.data_handler.update_symbol_data()
                # if any(self.period in s for s in self.granularities):
                #     period = self.granularities[self.period]
                # else:
                #     period = 15
                # schedule.every(period).seconds.do(self.handler.update_symbol_data)
                # schedule.run_pending()
                # time.sleep(1)

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
                        print('EVENT',event.type)
                        # print event.type
                        if event.type == 'MARKET':
                            self.strategy.calculate_signals(event)
                            self.portfolio.update_timeindex(event)

                        elif event.type == 'SIGNAL':
                            self.signals += 1
                            self.portfolio.update_signal(event)

                        elif event.type == 'ORDER':
                            self.orders += 1
                            self.execution_handler.execute_order(event)

                        elif event.type == 'FILL':
                            self.fills += 1
                            if self.simulation == False:
                                print("update_pos_from_broker")
                                self.portfolio.update_positions_from_broker(event)
                            elif self.simulation:
                                self.portfolio.update_fill(event)

            time.sleep(self.heartbeat)

    # def _output_performance(self):
    #     print("#BACKTEST # -", time.clock() - self.time0, "seconds process time")
    #     print("#BACKTEST # -", time.time() - self.time1, "seconds wall time")
    #
    #     """
    #     Outputs the strategy performance from the backtest.
    #     """
    #     self.portfolio.create_equity_curve_dataframe()
    #     print("Creating summary stats...")
    #     stats, plt = self.portfolio.output_summary_stats(self.period, self.data_handler.get_symbol_data(), self.s_file, length=self.length)
    #     print("Creating equity curve...")
    #     print(self.portfolio.equity_curve.tail(10))
    #     pprint.pprint(stats)
    #     print("Signals: %s" % self.signals)
    #     print("Orders: %s" % self.orders)
    #     print("Fills: %s" % self.fills)
    #     # plt.show()

    def trade(self):
        """
        Simulates the backtest and outputs performance.
        """
        ## todo should log errors

        try:
            self._set_up() ## todo e.g. ver dinheiro em cada exchange, ver posições abertas, começar a sacar dados,
        except Exception as e:
            print('"{}" error while setting up the trading environment.'.format(e))

        try:
            self._run()
        except Exception as e:
            print('"{}" runtime error while trading.'.format(e))

        try:
            self._tear_down()
        except Exception as e:
            print('"{}" error while cleaning the environment.'.format(e))

