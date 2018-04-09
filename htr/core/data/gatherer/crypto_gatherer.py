
import time
from datetime import datetime as dt
import zmq

class CryptoGatherer:
    def __init__(self, broker_handler):
        """.

        Args:
        	exchanges (list): List of exchanges

        """

        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind("tcp://127.0.0.1:5558")

        ## todo pass context?
        self.broker_handler = broker_handler('')
        self.errors = []
        self.last_tick = dt.now()

    def ticker(self, symbol):
        """Get ticks from exchanges."""

        try:
            ## Prevent more than one tick per second todo try to improve this
            while (dt.now() - self.last_tick).seconds < 1:
                continue

            tick = self.broker_handler.ticker(symbol.replace('/', ''))
            self.last_tick = dt.now()

        except Exception as e:
            print(e)
            self.errors.append(e.__str__() + symbol + str(dt.now()))
            ## todo maybe i dont want to exit
            return {}

        return tick

    def start_server(self):
        while True:
            try:
                request = str(self.socket.recv())
                # print("Request: ", request)
                time.sleep(1)
                request_type, symbol = request.split(' ')
                tick = self.ticker(symbol.replace('\'',''))

                print ("TICK TIMESTAMP - ", tick['timestamp'])
                if "tick" in request_type:
                    message = "{0} {1}".format(symbol, tick)
                    self.socket.send_string(message)

            except Exception as e:
                print(e)
                self.errors.append(e.__str__() + symbol + str(dt.now()))
                continue

# if __name__ == '__main__':
#     instruments = ['eur/USD', 'eur/JPY', 'USD/CAD']
#     stream = RateStreamer(instruments)
#
#     try:
#         t1 = Thread(None, stream.startStream,None,())
#         t1.start()
#         t2 = Thread(None, stream.startServer,None,())
#         t2.start()
#     except Exception as err:
#         print err
# from htr.core.brokerage import KrakenHandler
# c = CryptoGatherer(KrakenHandler)
# c.start_server()