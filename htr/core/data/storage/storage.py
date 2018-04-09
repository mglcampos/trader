from arctic import Arctic
import pandas as pd
from datetime import datetime as dt

class Artic(object):

    def __init__(self):
        self.artic = Arctic('localhost')
        ##print artic.list_libraries()
        if not any('forex.pairs' in s for s in self.artic.list_libraries()):
            self.artic.initialize_library('forex.pairs')

        self.pairs = self.artic['forex.pairs']
        self.results = self.artic['forex.results']

    def storeData(self, symbol, df):

        try:
            if any(symbol in s for s in self.pairs.list_symbols()):
                    self.pairs.append(symbol, df)
            else:
                    self.pairs.write(symbol, df)
        except Exception:
            print "COULDN'T STORE: invalid symbol or invalid dataframe"

        try:
            self.pairs.snapshot(str(df.tail(1).index))
        except Exception:
            print "SNAPSHOT ERROR: Snapshot already exists"
        print str(df.tail(1).index)
        # print self.pairs.read('eur/CAD').data.ix['2014, 1, 2':]

    def getData(self, symbol, start=None, end=dt.now(), Nrows=None):
        if start is None:
            if Nrows is None:
                try:
                    df = self.pairs.read(symbol).data
                    return df
                except Exception:
                    print "COULDN'T READ: invalid symbol"
            else:
                try:
                    df = self.pairs.read(symbol).data.tail(Nrows)
                    return df
                except Exception:
                    print "COULDN'T READ: invalid symbol or invalid Nrows"

        else:
            try:
                df = self.pairs.read(symbol).data.ix[start:end]
                return df
            except Exception:
                print "COULDN'T READ: invalid start/end dates or invalid symbol"



    def remove(self, symbol):
        try:
            self.pairs.delete(symbol)
        except Exception:
            print "COULDN'T REMOVE: invalid symbol"
        print("Erased: ",symbol)

    def getPairList(self):
        return self.pairs.list_symbols()

    def createDataframe(self, ask, bid, time):
        return pd.DataFrame({'ask': [ask], 'bid': [bid]}, [time])


# artic = Artic()
# df = pd.DataFrame({'prices': [1, 2, 3]},
#                           [dt(2014, 1, 4), dt(2014, 1, 5), dt(2014, 1, 6)])
#
# artic.storeData('eur/GBP', df)
# print artic.getPairList()
# print artic.getData('eur/CAD', Nrows=2)
# artic.remove('eur/GBP')
# print artic.getData('eur/GBP')