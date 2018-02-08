
import datetime
import time
import re

from htr.core.execution import ExecutionHandler
from htr.core.brokerage import KrakenHandler
from htr.core.events.event import FillEvent

class CryptoExecutionHandler(ExecutionHandler):
    """
   .
    """

    def __init__(self, context, events, broker_handler):
        """
        Initialises the handler, setting the event queues
        up internally.
        Parameters:
        events - The Queue of Event objects.
        """

        ##todo create a list of exchanges

        self.broker_handler = broker_handler

        self.order_id = 0
        self.events = events
        self.fill_dict = {}

    def create_fill(self, response, units, direction):
        """
        Handles the creation of the FillEvent that will be
        placed onto the events queue subsequent to an order
        being filled.
        """
        order_id = -1

        order_id = response['order_id']
        if not any(str(order_id) in str(s) for s in self.fill_dict):
            self.fill_dict[order_id] = {}

        fd = self.fill_dict[order_id]
        # # Prepare the fill data
        fd["symbol"] = response['symbol']
        fd['timestamp'] = response['timestamp']
        fd['price'] = response['price']
        fd['units'] = units
        fd['direction'] = direction
        fd['exchange'] = response['exchange']
        # Create a fill event object
        fill = FillEvent(
            datetime.datetime.utcnow(), fd["symbol"].replace('_', '/'),
            fd['exchange'], fd['units'], fd['direction'], None,
        )
        self.fill_dict[order_id]["filled"] = True
        # Place the fill event onto the event queue
        self.events.put(fill)

    def execute_order(self, event):
        """
        Parameters:
        event - Contains an Event object with order information.
        """

        ##TODO check if order went through, slippage, etc

        if event.type == 'ORDER':

            instrument = event.symbol.replace('/', '')
            print('Order instrument : ', instrument)
            order_type = event.order_type
            quantity = event.quantity
            direction = event.direction

            ##Close open position
            if direction == 'BUY' and order_type == 'MKT-CLOSE':
                ## todo check if quantity is right

                for i in range(0, 2):
                    try:
                        response = self.broker_handler.create_order(instrument, quantity, 0, direction.lower(), 'market')
                        self.create_fill(response, quantity, direction)
                        break

                    except Exception as e:
                        if re.search('Insufficient funds', e.__str__()) != None:
                            quantity = self.broker_handler.downsize_order(quantity)
                            print('Downsized {} order to {}.'.format(order_type, quantity))
                            continue

                        print('Failed to {} in {} order.'.format(direction, order_type))

            elif direction == 'SELL' and order_type == 'MKT-CLOSE':
                ## todo check if quantity is right
                for i in range(0, 2):
                    try:
                        response = self.broker_handler.create_order(instrument, quantity, 0, direction.lower(),
                                                                    'market')
                        self.create_fill(response, quantity, direction)
                        break

                    except Exception as e:
                        print('\nwtf exception', e)
                        if re.search('Insufficient funds', e.__str__()) != None:
                            quantity = self.broker_handler.downsize_order(quantity)
                            print('Downsized {} order to {}.'.format(order_type, quantity))
                            continue

                        print('Failed to {} in {} order.'.format(direction, order_type))


            elif direction == 'BUY' and order_type == 'MKT-OPEN':

                for i in range(0, 3):
                    ## todo check if quantity is right
                    try:
                        response = self.broker_handler.create_order(instrument, quantity, 0, direction.lower(), 'market')
                        self.create_fill(response, quantity, direction)
                        break

                    except Exception as e:
                        ## todo remove this
                        quantity = self.broker_handler.downsize_order(quantity)
                        print('Downsized {} order to {}.'.format(order_type, quantity))
                        continue

                        print('\nwtf exception', e)
                        # if re.search('Insufficient funds', e.__str__()) != None:
                        #     quantity = self.broker_handler.downsize_order(quantity)
                        #     print('Downsized {} order to {}.'.format(order_type, quantity))
                        #     continue

                    print('Failed to {} in {} order.'.format(direction, order_type))

            elif direction == 'SELL' and order_type == 'MKT-OPEN':
                ## todo check if quantity is right

                for i in range(0, 2):
                    ## todo check if quantity is right
                    try:
                        response = self.broker_handler.create_order(instrument, quantity, 0, direction.lower(),
                                                                    'market')

                        self.create_fill(response, quantity, direction)
                        break

                    except Exception as e:
                        if re.search('Insufficient funds', e.__str__()) != None:
                            quantity = self.broker_handler.downsize_order(quantity)
                            print('Downsized {} order to {}.'.format(order_type, quantity))
                            continue

                    print('Failed to {} in {} order.'.format(direction, order_type))

            time.sleep(1)
            # Increment the order ID for this session
            self.order_id += 1