
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


class LiveTrader:
    """
    Event-driven backtest.
    """

    def __init__(self, data_handler, execution_handler, portfolio, strategy, risk_handler, broker_handler, context):
        """
        .
        Parameters:

        """

        self.symbol_list = context.symbol_list
        self.context = context
        self.heartbeat = context.heartbeat
        self.data_handler_cls = data_handler
        self.execution_handler_cls = execution_handler
        self.portfolio_cls = portfolio
        self.strategy_cls = strategy
        self.risk_handler_cls = risk_handler
        self.broker_handler_cls = broker_handler
        self.events = queue.Queue()
        self.signals = 0
        self.orders = 0
        self.fills = 0
        self.num_strats = 1

    def _set_up(self):
        """
        Generates the trading instance objects from
        their classes and sets up the trading environment.
        """


        self.broker_handler = self.broker_handler_cls(self.context)

        self.initial_capital = self.broker_handler.get_cash(symbol=self.context.base_currency)
        setattr(self.context, 'initial_capital', self.initial_capital)

        self.data_handler = self.data_handler_cls(self.context, self.events)
        self.strategy = self.strategy_cls(self.context, self.events, self.data_handler)
        self.risk_handler = self.risk_handler_cls(self.context)
        ## todo get open positions, set them in context so porfolio inits holdings
        self.portfolio = self.portfolio_cls(self.context, self.events, self.risk_handler, self.data_handler, self.broker_handler)
        self.execution_handler = self.execution_handler_cls(self.context, self.events, self.broker_handler)


    def _run(self):
        """
        Executes the backtest.
        """

        i = 0
        while True:
            i += 1
            print(i)

            # Request tick.
            self.data_handler.update_symbol_data()

            # Handle the events
            while True:
                try:
                    event = self.events.get(False)

                except queue.Empty:
                    break

                else:
                    if event is not None:
                        print('\nEVENT :',event.type)
                        if event.type == 'MARKET':
                            self.strategy.calculate_signals()
                            self.portfolio.update_timeindex(event)

                        elif event.type == 'SIGNAL':
                            self.signals += 1
                            self.portfolio.process_signal(event)

                        elif event.type == 'ORDER':
                            self.orders += 1
                            self.execution_handler.execute_order(event)

                        elif event.type == 'FILL':
                            self.fills += 1
                            self.portfolio.update_fill(event)

            time.sleep(self.heartbeat)

    def trade(self):
        """
        Starts trading.
        """
        ## todo should log errors

        try:
            self._set_up() ## todo e.g. ver dinheiro em cada exchange, ver posições abertas
        except Exception as e:
            print('\n"{}" error while setting up the trading environment.'.format(e))
            return

        try:
            self._run()
        except Exception as e:
            print('\n"{}" runtime error while trading.'.format(e))
            raise e

        # try:
        #     self._tear_down()
        # except Exception as e:
        #     print('"{}" error while cleaning the environment.'.format(e))

