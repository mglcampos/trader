
from htr.core.execution.execution import ExecutionHandler
from htr.core.events.event import OrderEvent

class CryptoExecutionHandler(ExecutionHandler):
    """
    The simulated execution handler simply converts all order
    objects into their equivalent fill objects automatically
    without latency, slippage or fill-ratio issues.
    This allows a straightforward "first go" test of any strategy,
    before implementation with a more sophisticated execution
    handler.
    """

    def __init__(self, events, oanda=None):
        """
        Initialises the handler, setting the event queues
        up internally.
        Parameters:
        events - The Queue of Event objects.
        """
        if oanda != None:
            self.oanda = oanda
        else:
            self.oanda = BrokerOanda()

        self.order_id = 0
        self.events = events
        self.fill_dict = {}

    def create_order(self, symbol, order_type, quantity, direction):

        order = OrderEvent(symbol, order_type, quantity, direction)

        return order


    def create_fill(self, response, fill_type, units, direction):
        """
        Handles the creation of the FillEvent that will be
        placed onto the events queue subsequent to an order
        being filled.
        """
        trade_id = -1
        if fill_type == 'OPEN':
            trade = response.get('tradeOpened')
            trade_id = trade.get('id')
            if not any(str(trade_id) in str(s) for s in self.fill_dict):
                self.fill_dict[trade_id] = {}
            fd = self.fill_dict[trade_id]
            # Prepare the fill data
            fd["symbol"] = response.get('instrument')
            fd['time'] = response.get('time')
            fd['price'] = response.get('price')
            fd['units'] = units
            fd['direction'] = direction
            fd['takeProfit'] = trade.get('takeProfit')
            fd['stopLoss'] = trade.get('stopLoss')

             ##TODO exchange is not being used properly
            fd['exchange'] = 'OPEN'
            # Create a fill event object
            fill = FillEvent(
                datetime.datetime.utcnow(), fd["symbol"].replace('_','/'),
                fd['exchange'], fd['units'], fd['direction'], None, broker='OANDA'
            )
            self.fill_dict[trade_id]["filled"] = True
            # Place the fill event onto the event queue
            self.events.put(fill)

        elif fill_type == 'Close':
            trade_id = response.get('id')
            fd = self.fill_dict[trade_id]
            fd['exchange'] = 'Close'
            fd['price'] = response.get('price')
            fd['direction'] = response.get('side')
            fd['profit'] = response.get('profit')
            fd["symbol"] = response.get('instrument')
            fd['units'] = units
            print('creating fill for',response.get('instrument'))
            print('fill direction ', response.get('side'))
            fill = FillEvent(
                datetime.datetime.utcnow(), fd["symbol"],
                fd['exchange'], fd['units'], fd['direction'], None
            )
            self.fill_dict[trade_id]["filled"] = True
            # Place the fill event onto the event queue
            self.events.put(fill)
        # additional fills.



    def execute_order(self, event):
        """
        Parameters:
        event - Contains an Event object with order information.
        """
        if event.type == 'ORDER':
            # Prepare the parameters for the asset order
            instrument = event.symbol
            # asset_type = "STK"
            order_type = event.order_type
            quantity = event.quantity
            direction = event.direction
            print('quantity', quantity)
            print('direction', direction)
            ##TODO alterar para permitir mais do que uma posicao por instrumento
            ##Close open position
            if direction == 'BUY' and event.order_type == 'MKT-CLOSE':
                open_pos = self.oanda.get_trades(instrument = event.symbol)
                print (open_pos)
                trades = open_pos.get('trades')
                trade_id = trades[0].get('id')

                trade = self.oanda.get_trades(trade_ids = [trade_id])
                print (trade)
                units = trade.get('trades')[0].get('units')
                trade_closed = self.oanda.close_trade(trade_id)
                print("Trade Closed - ", trade_closed)
                self.create_fill(trade_closed, 'CLOSE', units, direction)

            elif direction == 'SELL' and event.order_type == 'MKT-CLOSE':
                open_pos = self.oanda.get_trades(instrument = event.symbol)
                trades = open_pos.get('trades')
                print("sell-mktclose",trades)
                trade_id = trades[0].get('id')
                print(open_pos)
                trade = self.oanda.get_trades(trade_ids=[trade_id])
                units = trade.get('trades')[0].get('units')
                trade_closed = self.oanda.close_trade(trade_id)
                print("Trade Closed - ", trade_closed)
                self.create_fill(trade_closed, 'CLOSE', units, direction)

            ##TODO reduzir para um elif
            elif direction == 'BUY' and event.order_type == 'MKT-OPEN':
                print('enter buying')
                oanda_order = self.oanda.create_long_order(instrument, quantity)
                ##TODO check if order went through, slippage, etc
                print(oanda_order)
                self.create_fill(oanda_order, 'OPEN', quantity, direction)

            elif direction == 'SELL' and event.order_type == 'MKT-OPEN':
                print('enter selling')
                oanda_order = self.oanda.create_short_order(instrument, quantity)
                ##TODO check if order went through, slippage, etc
                print(oanda_order)
                self.create_fill(oanda_order, 'OPEN', quantity, direction)


            time.sleep(1)
            # Increment the order ID for this session
            self.order_id += 1