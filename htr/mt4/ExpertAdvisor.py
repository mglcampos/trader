import datetime
import time
import zmq_client
import pandas as pd

class ExpertAdvisor():
    def __init__(self):
        context = zmq_client.Context()
        self.sub = context.socket(zmq_client.SUB)
        self.sub.setsockopt(zmq_client.SUBSCRIBE, '')
        self.sub.bind("tcp://127.0.0.1:2027")
        self.pub = context.socket(zmq_client.PUB)
        self.pub.bind("tcp://127.0.0.1:2028")

        self.cache = {}

    def listen(self):
        values = []
        while True:
            msg = self.sub.recv()
            channel, msg = msg.split('|', 1)
            if channel == 'tick':
                print(" msg - " + msg)
                values = msg.split(' ')
                now = datetime.datetime.now()
                time = datetime.datetime.strptime(values[3], '%H:%M:%S').replace(year=now.year, month=now.month, day=now.day)
                ask = values[1]
                bid = values[0]
                symbol = values[2][:3] + '/' + values[2][3:]
                print(" ask - " + ask)
                print(" bid - " + bid)
                print(" symbol - " + symbol)
                print(" time - " + str(time))
                if symbol in self.cache.keys():

                    ## remove this if to enable more than one value per second
                    if time > self.cache[symbol]['time']:
                        self.cache[symbol]['df'] = self.cache[symbol]['df'].append(pd.DataFrame({'ask': [ask], 'bid': [bid]}, [time]))
                        self.cache[symbol]['time'] = time
                        print(self.cache[symbol]['df'])
                else:
                    self.cache[symbol] = {'df' : pd.DataFrame({'ask': [ask], 'bid': [bid]}, [time]), 'time' : time}


if __name__ == '__main__':
    expert = ExpertAdvisor()
    expert.listen()