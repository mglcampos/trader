from OandaWrapper import Streamer, OandaFormat
from thread import start_new_thread
from threading import Thread
import time
import zmq
import pprint
from datetime import datetime
from DataStorage import Artic
import dateutil.parser


class RateStreamer(Streamer):
    def __init__(self, instruments, environment='practice'):
        self.OANDA_ACCESS_TOKEN = "970b08eac223cc0fb6b9d50aa9bd7bd5-aa2497c02bec0264cc32c98909be36fd"
        self.OANDA_ACCOUNT_ID = 6068438
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind("tcp://127.0.0.1:5558")
        self.instruments = instruments
        self.cache = {}
        self.last_date = {}
        for s in self.instruments:
            self.cache[s] = []
            self.last_date[s] = datetime.now()
        Streamer.__init__(self, access_token=self.OANDA_ACCESS_TOKEN, environment=environment)


    def startStream(self):
        # Format instructions to oanda format
        formInstru = OandaFormat().formatInstruments(self.instruments)
        # Start streaming and call on_success function
        Streamer.start(self, instruments=formInstru, accountId=self.OANDA_ACCOUNT_ID, ignore_heartbeat=True)

    def on_success(self, data):

        data = data.get('tick')
        symbol = data.get('instrument').replace('_', '/', 1)
        time = dateutil.parser.parse(data.get('time'))
        bid = data.get('bid')
        ask = data.get('ask')
        self.cache[symbol].append({'ask': ask, 'bid': bid, 'time': time})
        print('CACHE:', symbol, len(self.cache[symbol]), self.cache[symbol][-1:])

        if len(self.cache[symbol]) > 499:
            try:
                self.artic = Artic()
                df = self.artic.createDataframe(ask, bid, time)
                print("Storing data for ", symbol)
                self.artic.storeData(symbol, df)
                print("Cleaning cache ", symbol)
                self.cache[symbol] = []
            except Exception as e:
                print(e)


    def startServer(self):
        while True:
            request = self.socket.recv()
            print("Request: ", request)
            time.sleep(1)
            request, symbol = request.split(' ')
            if len(self.cache[symbol])> 1:
                print ("Before TICKKKKKKKKK - ", self.cache[symbol][-1:][0]['time'])

                try:   ##TODO change this to identify repeated
                    tick_date = self.cache[symbol][-1:][0]['time'].replace(tzinfo=None)
                except Exception:
                    tick_date = self.cache[symbol][-1:][0]['time']

                print ("TICKKKKKKKKK - ", tick_date)
                print ("NOWWWWWWWWWW - ", self.last_date[symbol])
                if self.last_date[symbol] < tick_date:
                    self.last_date[symbol] = tick_date
                    if request == "tick":
                        data = dict(self.cache[symbol][-1:][0])
                        data['time'] = str(data['time'])
                        message = "{0} {1}".format(symbol, data)
                        self.socket.send(message)
                else:
                    message = 'Nothing New'
                    self.socket.send(message)
            else:
                message = 'Nothing New'  ##TODO maybe change this
                self.socket.send(message)


    def disconnect(self):
        # self.connected = False
        Streamer.disconnect()



if __name__ == '__main__':
    instruments = ['EUR/USD', 'EUR/JPY', 'USD/CAD']
    stream = RateStreamer(instruments)

    try:
        t1 = Thread(None, stream.startStream,None,())
        t1.start()
        t2 = Thread(None, stream.startServer,None,())
        t2.start()
    except Exception as err:
        print err