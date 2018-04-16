# from __future__ import print_function
#
# import zmq
# import time
# from random import choice
# from random import randrange
# import zmq
# from datetime import datetime
# from matplotlib.dates import date2num
#
#
# context = zmq.Context()
# socket = context.socket(zmq.SUB)
#
# socket.connect("tcp://127.0.0.1:2027")
#
# socket.setsockopt(zmq.SUBSCRIBE,"tick")
#
#
# while True:
#     string = socket.recv()
#     print(string + " " + str(date2num(datetime.now())))
#     print(len(string))


# def result_collector():
#
#     context = zmq.Context()
#     socket = context.socket(zmq.SUB)
#     socket.setsockopt(zmq.SUBSCRIBE, 'USD_CAD ')
#     socket.connect("tcp://127.0.0.1:5558")
#     while True:
#         msg = socket.recv()
#         print(msg)
#
# result_collector()
#
# def publisher():
#     stock_symbols = ['RAX', 'EMC', 'GOOG', 'AAPL', 'RHAT', 'AMZN']
#
#     context = zmq.Context()
#     socket = context.socket(zmq.PUB)
#     socket.bind("tcp://127.0.0.1:4999")
#
#     while True:
#         time.sleep(3)
#         # pick a random stock symbol
#         stock_symbol = choice(stock_symbols)
#         # set a random stock price
#         stock_price = randrange(1, 100)
#
#         # compose the message
#         msg = "{0} ${1}".format(stock_symbol, stock_price)
#
#         print("Sending Message: {0}".format(msg))
#
#         # send the message
#         socket.send(msg)
#         # Python3 Note: Use the below line and comment
#         # the above line out
#         # socket.send_string(msg)
#
# def subscriber():
#     context = zmq.Context()
#     socket = context.socket(zmq.SUB)
#     socket.setsockopt(zmq.SUBSCRIBE, '')
#     socket.bind("tcp://127.0.0.1:2027")
#
#     while True:
#         msg = socket.recv()
#         # Python3 Note: Use the below line and comment
#         # the above line out
#         # msg = socket.recv_string()
#         print(" msg - " + msg)

#
# subscriber()

# publisher()
# subscriber()

# import sys
# import zmq
#
# port = "2027"
#
#
# # Socket to talk to server
# context = zmq.Context()
# socket = context.socket(zmq.SUB)
#
# socket.connect("tcp://localhost:%s" % port)
#
# # Subscribe to zipcode, default is NYC, 10001
# topicfilter = "tick"
# socket.setsockopt(zmq.SUBSCRIBE, topicfilter)
#
# # Process 5 updates
# total_value = 0
# for update_nbr in range(5):
#     string = socket.recv()
#     topic, messagedata = string.split()
#     total_value += int(messagedata)
#     print topic, messagedata
#
# print "Average messagedata value for topic '%s' was %dF" % (topicfilter, total_value / update_nbr)
#
#
#
#
#
#






# import zmq
    # import random
    # import sys
    # import time
    #
    # port = "2027"
    # if len(sys.argv) > 1:
    #     port =  sys.argv[1]
    #     int(port)
    #
    # context = zmq.Context()
    # socket = context.socket(zmq.PUB)
    # socket.bind("tcp://*:%s" % port)
    # while True:
    #     topic = random.randrange(9999,10005)
    #     messagedata = random.randrange(1,215) - 80
    #     print "%d %d" % (topic, messagedata)
    #     socket.send("%d %d" % (topic, messagedata))
    #     time.sleep(1)


# compArray[0] = TRADE
# compArray[1] = ACTION(e.g.OPEN, MODIFY, CLOSE)
# compArray[2] = TYPE(e.g.OP_BUY, OP_SELL, etc - only
# used
# when
# ACTION = OPEN)
#
# // ORDER
# TYPES:
# // https: // docs.mql4.com / constants / tradingconstants / orderproperties
#
# // OP_BUY = 0
# // OP_SELL = 1
# // OP_BUYLIMIT = 2
# // OP_SELLLIMIT = 3
# // OP_BUYSTOP = 4
# // OP_SELLSTOP = 5
#
# compArray[3] = Symbol(e.g.EURUSD, etc.)
# compArray[4] = Open / Close
# Price(ignored if ACTION = MODIFY)
# compArray[5] = SL
# compArray[6] = TP
# compArray[7] = lOTS
# compArray[8] = comments / ticket

import zmq
# Sample Commands for ZeroMQ MT4 EA
eurusd_buy_order = "TRADE|OPEN|0|EURUSD|0|50|50|0.01|Python-to-MT4"
eurusd_sell_order = "TRADE|OPEN|1|EURUSD|0|50|50|0.01|Python-to-MT4"
eurusd_closebuy_order = "TRADE|CLOSE|0|EURUSD|0|50|50|0.01"
get_rates = "RATES|BTCUSD"

# Sample Function for Client
def zeromq_mt4_ea():

    # Create ZMQ Context
    context = zmq.Context()

    # Create REQ Socket
    reqSocket = context.socket(zmq.REQ)
    reqSocket.connect("tcp://localhost:5555")

    # Create PULL Socket
    pullSocket = context.socket(zmq.PULL)
    pullSocket.connect("tcp://localhost:5556")

    # Send RATES command to ZeroMQ MT4 EA
    remote_send(reqSocket, get_rates)
    # PULL from pullSocket
    remote_pull(pullSocket)

    # Send BUY EURUSD command to ZeroMQ MT4 EA
    remote_send(reqSocket, eurusd_buy_order)
    ticket = remote_pull(pullSocket).split('|', 1)[1]

    # Send CLOSE EURUSD command to ZeroMQ MT4 EA. You'll need to append the
    # trade's ORDER ID to the end, as below for example:
    remote_send(reqSocket, eurusd_closebuy_order + "|" + ticket)

    # PULL from pullSocket
    remote_pull(pullSocket)

# Function to send commands to ZeroMQ MT4 EA
def remote_send(socket, data):

    try:
        socket.send_string(data)
        msg = socket.recv_string()
        print("SENT: ", msg)

    except zmq.Again as e:
        print("Waiting for PUSH from MetaTrader 4..")

# Function to retrieve data from ZeroMQ MT4 EA
def remote_pull(socket):

    try:
        # msg = socket.recv(flags=zmq.NOBLOCK)
        msg = socket.recv(flags=zmq.NOBLOCK)
        print("RECEIVED: ", msg)
        return msg

    except zmq.Again as e:
        print("Waiting for PUSH from MetaTrader 4..")

# Run Tests
zeromq_mt4_ea()
