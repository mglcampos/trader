
import time
import zmq

from htr.helpers.wrappers import Bitfinex, BitfinexV2


class CryptoGatherer:
    def __init__(self, exchanges=None, environment='practice'):
        """.

        Args:
        	exchanges (list): List of exchanges

        """
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind("tcp://127.0.0.1:5558")

        self.bitfinex = Bitfinex.Public()
        print(self.bitfinex.ticker())

    def ticker(self, symbol):
        """Get ticks from exchanges."""
        print(symbol)
        ## todo point of abstraction to the exchange, change this to implement sockets or http stream and more than one exchange
        tick = self.bitfinex.ticker(symbol)

        return tick

    def startServer(self):
        while True:
            request = str(self.socket.recv())
            print("Request: ", request)
            time.sleep(1)
            request_type, symbol = request.split(' ')
            print(request_type)
            tick = self.ticker(symbol.replace('\'',''))
            print ("TICK TIMESTAMP - ", tick['timestamp'])
            if "tick" in request_type:
                message = "{0} {1}".format(symbol, tick)
                self.socket.send_string(message)


# if __name__ == '__main__':
#     instruments = ['EUR/USD', 'EUR/JPY', 'USD/CAD']
#     stream = RateStreamer(instruments)
#
#     try:
#         t1 = Thread(None, stream.startStream,None,())
#         t1.start()
#         t2 = Thread(None, stream.startServer,None,())
#         t2.start()
#     except Exception as err:
#         print err

c = CryptoGatherer()
c.startServer()