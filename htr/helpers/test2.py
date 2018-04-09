import datetime
import OandaWrapper
import schedule
import time
import zmq

OANDA_ACCESS_TOKEN = "970b08eac223cc0fb6b9d50aa9bd7bd5-aa2497c02bec0264cc32c98909be36fd"
OANDA_ACCOUNT_ID = 6068438  # put your access id here

def main():
    # print "------ System online -------", datetime.now()
    oanda = OandaWrapper.API(environment="practice",
                             access_token=OANDA_ACCESS_TOKEN)

    # print oanda.get_accounts()
    start = datetime.datetime(2014, 1, 1)
    end = datetime.datetime(2016, 1, 27)
    print oanda.get_history(instrument="EUR_USD", granularity="M1")

    # response = oanda.get_prices(instruments="EUR_USD")
    # prices = response.get("prices")
    # buy_price = prices[0].get("bid")
    #
    # print "Buy at", buy_price

    # trade_id = oanda.create_order(OANDA_ACCOUNT_ID, instrument="EUR_USD",
    #                               units=1000,
    #                               side='buy',
    #                               type='market')
    #
    # print "Trading id", trade_id


def send():
    symbol = 'eur/USD'
    request = 'tick'
    message = "{0} {1}".format(request, symbol)
    print("send_message - ", message)
    socket.send(message)
    response = socket.recv()
    print "Received reply ", request, "[", response, "]"


if __name__ == "__main__":
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    # socket.setsockopt(zmq.SUBSCRIBE, '')
    socket.connect("tcp://127.0.0.1:5558")
    request = 'tick'
    # send()
    schedule.every(5).seconds.do(send)
    while True:
        schedule.run_pending()
        time.sleep(1)





    # print("Teste1")
    # while True:
    #     schedule.every(1).seconds.do(main)
    #     schedule.run_pending()
    #     time.sleep(1)
    #     print("Teste2")
    # print("Teste12")
    # # main()  # from datetime import datetime
# import pandas.io.data as web
# from numpy import cumsum, log, polyfit, sqrt, std, subtract
# from numpy.random import randn
# # Download the Amazon OHLCV data from 1/1/2000 to 1/1/2015
# amzn = web.DataReader("AMZN", "yahoo", datetime(2000,1,1), datetime(2015,1,1))
# def hurst(ts):
#     """Returns the Hurst Exponent of the time series vector ts"""
#     # Create the range of lag values
#     lags = range(2, 100)
#     # Calculate the array of the variances of the lagged differences
#     tau = [sqrt(std(subtract(ts[lag:], ts[:-lag]))) for lag in lags]
#     # Use a linear fit to estimate the Hurst Exponent
#     poly = polyfit(log(lags), log(tau), 1)
#     # Return the Hurst exponent from the polyfit reports
#     return poly[0]*2.0
# # Create a Gometric Brownian Motion, Mean-Reverting and Trending Series
# gbm = log(cumsum(randn(100000))+1000)
# mr = log(randn(100000)+1000)
# tr = log(cumsum(randn(100000)+1)+1000)
# # Output the Hurst Exponent for each of the above series
# # and the price of Amazon (the Adjusted Close price) for
# # the ADF test given above in the article
# print("Hurst(GBM): %s" % hurst(gbm))
# print("Hurst(MR): %s" % hurst(mr))
# print("Hurst(TR): %s" % hurst(tr))
#
# print("Hurst(AMZN): %s" % hurst(amzn['Adj Close']))
