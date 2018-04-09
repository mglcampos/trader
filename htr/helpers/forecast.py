from __future__ import print_function

# import json
import time
import datetime
import numpy as np
import psutil
import json
import pandas as pd
from talib.abstract import *
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.lda import LDA
from sklearn.metrics import confusion_matrix, recall_score
from sklearn.qda import QDA
from sklearn.svm import LinearSVC, SVC
from sklearn.naive_bayes import GaussianNB,  MultinomialNB
from sklearn.feature_selection import RFE
from sklearn.decomposition import PCA
from sklearn.svm import SVR
from pymongo import MongoClient
# from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.api as sm
# from Playground import getKalmanFilter
# import matplotlib.pyplot as plt
# import itertools
# import operator

class Forecast(object):


    def create_lagged_series(self, symbol_list, start_date=None, end_date=None, lags=5, s_file=None):

        t0 = time.clock()
        t1 = time.time()

        # Obtain stock information from Yahoo Finance
        # ts = DataReader(
        #     symbol, "yahoo",
        #     start_date-datetime.timedelta(days=365),
        #     end_date
        # )
        symbol_data = self.getDataframe(symbol_list, s_file=s_file)
        lagged_data = {}
        for s in symbol_data:           
            if start_date == None and end_date == None:
                start_date = datetime.datetime.strptime(symbol_data[s]['Day'][1], '%Y.%m.%d')
                print('start_date', start_date)
                end_date = datetime.datetime.strptime(symbol_data[s]['Day'][len(symbol_data[s]['Day']) - 1], '%Y.%m.%d')
                print('end_date', end_date)
            # print(symbol_data[s].index)
            # print(symbol_data[s], "symbol_data[s]")
            # Create the new lagged DataFrame
            tslag = pd.DataFrame(index=symbol_data[s].index)
    
            tslag["Today"] = symbol_data[s]["Close"]
    
            tsret = pd.DataFrame(index=tslag.index)
    
            for column in symbol_data[s].columns.values:
                if column not in ['Day', 'Open', 'High', 'Low', 'Close']:
                    tsret[column] = symbol_data[s][column].shift(1)
                ##todo check if shifts the ts dataframe too

            # Create the shifted lag series of prior trading period close values
            for i in range(0, lags):
                # tslag["Lag%s" % str(i+1)] = symbol_data[s]["Adj Close"].shift(i+1)
                tslag["Lag%s" % str(i + 1)] = symbol_data[s]["Close"].shift(i + 1)
                tslag["Low_Lag%s" % str(i + 1)] = symbol_data[s]["Low"].shift(i + 1)
                tslag["High_Lag%s" % str(i + 1)] = symbol_data[s]["High"].shift(i + 1)
                tslag["Open_Lag%s" % str(i + 1)] = symbol_data[s]["Open"].shift(i + 1)

            tsret["Day"] = symbol_data[s]["Day"]
    
            tsret["Today"] = tslag["Today"].pct_change()*100.0
            # If any of the values of percentage returns equal zero, set them to
            # a small number (stops issues with QDA model in Scikit-Learn)
    
            for i,x in enumerate(tsret["Today"]):
                if (abs(x) < 0.0001):
                    tsret["Today"][i] = 0.0001
    
            # Create the lagged percentage returns columns
            for i in range(0, lags):
    
                tsret["Lag%s" % str(i+1)] = \
                tslag["Lag%s" % str(i+1)].pct_change()*100.0
                tsret["Low_Lag%s" % str(i + 1)] = \
                tslag["Low_Lag%s" % str(i + 1)].pct_change() * 100.0
                tsret["High_Lag%s" % str(i + 1)] = \
                tslag["High_Lag%s" % str(i + 1)].pct_change() * 100.0
                tsret["Open_Lag%s" % str(i + 1)] = \
                tslag["Open_Lag%s" % str(i + 1)].pct_change() * 100.0
            # Create the "Direction" column (+1 or -1) indicating an up/down day
            tsret["Direction"] = np.sign(tsret["Today"])

            tsret = tsret.dropna(axis=0)
            print('\npct_change()')
            # print(tsret['Lag3'][200:].head())
            # for ir in tsret.itertuples():
            #     date = tsret.loc[ir[0],'Day'].split('.')
            #     tsret.loc[ir[0], 'Day'] = datetime.datetime(int(date[0]),int(date[1]),int(date[2]))
            tsret['Day'] = pd.to_datetime(tsret['Day'])
            tsret = tsret[tsret['Day'] >= start_date]

            for column in tsret.columns.values:
                if column not in ['Day', 'Direction']:
                    symbol = s.replace('/', '')
                    tsret[symbol + column] = self.normalize_data(tsret[column])
                    del tsret[column]
            print('\nNORMALIZED')
            print(tsret[200:].head())
            ##todo boundery hardcoded, change to dropna
            lagged_data[s] = tsret[200:]

        print("# create_lagged # -", time.clock() - t0, "seconds process time")
        print("# create_lagged # -", time.time() - t1, "seconds wall time")
        # return tsret[lags:]
        return lagged_data

    def add_features(self, df):

        t0 = time.clock()
        t1 = time.time()

        inputs = {
            'open': df['Open'],
            'high': df['High'],
            'low': df['Low'],
            'close': df['Close']
        }

        # for t in [3, 11, 20, 31, 50, 100, 200]:
        #     df["SMA%s" % str(t)] = SMA(inputs, timeperiod=t)
        #     df["SMA%s" % str(t)] = ( df['Close'] / df["SMA%s" % str(t)] ) - 1
        # for t in [3, 4, 5, 8, 9, 10]:
        #     df["Momentum%s" % str(t)] = MOM(inputs, timeperiod=t)
        # for t in [9, 15, 30]:
        #     macd, macdsignal, macdhist = MACD(inputs, fastperiod=12, slowperiod=26, signalperiod=t)  # 15, 30
        #     df["MACD%s" % str(t)] = macd
        # for t in [3, 11, 20, 31, 50, 100, 200]:
        #     df["EMA%s" % str(t)] = EMA(inputs, timeperiod=t)
        #     df["EMA%s" % str(t)] = (df['Close'] / df["EMA%s" % str(t)]) - 1
        # for t in [6, 7, 8, 9, 10]:
        #     df["WILLIAMS%s" % str(t)] = WILLR(inputs, timeperiod=t)
        # for t in [5, 12, 13, 14, 15]:
        #     df["ROCP%s" % str(t)] = ROCP(inputs, timeperiod=t)
        # for t in [3, 5, 8, 11, 13, 14, 17, 20, 31]:
        #     df["ATR%s" % str(t)] = ATR(inputs, timeperiod=t)
        #     df["RSI%s" % str(t)] = RSI(inputs, timeperiod=t)
        #     df["ADX%s" % str(t)] = ADX(inputs, timeperiod=t)
        # for t in [3, 4, 5, 10, 20, 30, 50]:
        #     df["BETA%s" % str(t)] = BETA(inputs, timeperiod=t)
        # for t in [20]:
        #     df["BOLLINGER%s" % str(t)] = (df['Close'].values - pd.rolling_mean(df['Close'], window=t)) / (2 *  pd.rolling_std(df['Close'], window=t))

        # df['KALMAN'] = getKalmanFilter(df)
        # slowk, slowd = STOCH(inputs, fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
        # df['STOCHWK'] = slowk
        # df['STOCHWD'] = slowd
        # df['SIN'] = SIN(inputs)
        # df['CCI'] = CCI(inputs, timeperiod=14)


        print("# add_features # -", time.clock() - t0, "seconds process time")
        print("# add_features # -", time.time() - t1, "seconds wall time")

        # print(df[-200:].head())

        return df

    def getDataframe(self, symbol_list, s_file='_H1_2012'):

        t0 = time.clock()
        t1 = time.time()
        symbol_data = {}
        for symbol in symbol_list:
            s = symbol.replace('/','')
            # symbol_data[symbol] = pd.read_csv("histdata/" + s + s_file,
            #                  names=['Type','Day', 'Time', 'Open', 'High', 'Low', 'Close'] )
            symbol_data[symbol] = pd.read_csv("histdata/" + s + s_file, names=['Type', 'Day', 'Time', 'Open', 'High', 'Low', 'Close'], usecols=['Day', 'Open', 'High', 'Low', 'Close'])
            print('len_dataset', symbol, len(symbol_data[symbol]))
            symbol_data[symbol] = symbol_data[symbol].reset_index(drop=True)
            symbol_data[symbol] = self.add_features(symbol_data[symbol])


        


        print("# get_dataframe # -", time.clock() - t0, "seconds process time")
        print("# get_dataframe # -", time.time() - t1, "seconds wall time")

        # print(symbol_data)

        return symbol_data

    def normalize_data(self, c):
        # print('len_before_normalize',len(c))
        # print('len_after_normalize', len(normalize(c, axis=1)))
        # return normalize(c, axis=1)
        # return (c - np.mean(c)) / np.std(c)
        min = np.min(c)
        max = np.max(c)
        return (c - min) / (max - min)

    def feature_extraction(self, X, y, allfeatures, features_len=5):

        # allfeatures = ['EURNZDLag3','EURAUDATR20','EURAUDATR31','EURAUDOpen_Lag1','EURAUDLag2','Day']
        estimator = SVR(kernel="linear")
        selector = RFE(estimator, features_len, step=1)
        print('feature_extraction')
        # X = X.dropna(axis=1)
        X = X[np.isfinite(X)]
        print('len_fselector_datasets',len(X[allfeatures]),len(y['Direction']))
        selector = selector.fit(X[allfeatures][:-1], y['Direction'][:-1])
        # print('selector_support',selector.support_)
        print('selector_ranking',selector.ranking_)
        print('zip_ranking', zip(allfeatures,selector.ranking_))
        print('selected', [i for i in zip(allfeatures,selector.ranking_) if i[1] == 1])
        return [i[0] for i in zip(allfeatures,selector.ranking_) if i[1] == 1]

    def runForecast(self, lags=50, features_len=10, s_file='_M1_2012_2013', rfe = False, symbol_list = ['eur/USD'], pca_reduction=False, store=True):

        print('rfe', rfe)
        print('pca', pca_reduction)

        symbol_list = symbol_list
        corr = []
        # Create a lagged series
        series = self.create_lagged_series(symbol_list, lags=lags, s_file=s_file)
        snpret = series[symbol_list[0]]

        print(snpret.head())

        subsets = []
        mainindicators = []
        corrindicators = []

        for column in snpret.columns.values:
            if column not in ['Direction', symbol_list[0].replace('/','') + 'Today']:
                mainindicators.append(column)

        if len(symbol_list) > 1:
            corr = series[symbol_list[1]]
            print(corr.head())
            for column in corr.columns.values:
                if column not in ['Day', 'Direction', symbol_list[1].replace('/','') +  'Today']:
                    corrindicators.append(column)


        # for i in range(1,4):
        #     for pair in list(itertools.combinations(indicators, i)):
        #        subsets.append(['Day'] + [x for x in pair]) ##groups of indicators
        #        subsets.append(['Day'] + [x for x in pair] + prices) ##groups of indicators + price
        #        subsets.append(['Day'] + [x for x in pair] + ['Lag1','Lag2','Lag3','Lag4','Lag5'])  ##groups of indicators + price

        # subsets.append(prices + ['Day', 'MACD']) ##all indicators
        # subsets.append(['SMA20','SMA50', 'SMA100', 'SMA200','Day','Bollinger'] ) ##all indicators + close price
        # subsets.append(['SMA20','ERSI20','Day','Bollinger']) ##all subset of 4 indicators + close price
        # subsets.append(prices + indicators + ['Day']) ##everything
        subsets.append(mainindicators + corrindicators)
        # subsets.append(['EURNZDLag3','EURAUDATR20','EURAUDATR31','EURAUDOpen_Lag1','EURAUDLag2','Day'])

        scores = {}
        bestscores = {}

        for subset in subsets:
            print('original_subset', subset)
            print('original_len_subset', len(subset))
            bestscores['len_subsets'] = len(subsets)
            print('len_subsets',bestscores['len_subsets'])
            if len(corr) > 1:
                X = pd.concat([snpret[mainindicators][:-1], corr[corrindicators][:-1]], axis=1)
            else:
                X = snpret[mainindicators][:-1]
            # X = snpret[subset][:-1]
            y = snpret[["Direction", 'Day']][:-1]
            # The test data is split into two parts: Before and after at 3/4
            half = (len(snpret) / 4) * 3
            start_test = snpret['Day'][half]
            print('\nStart_Test ', start_test)
            X_train = X[X['Day'] < start_test]
            X_test = X[X['Day']  >= start_test]
            y_train = y[y['Day']  < start_test]
            y_test = y[y['Day']  >= start_test]
            subset.remove('Day')

            X_train = X_train[subset]
            X_test = X_test[subset]
            ##dimension reduction
            if rfe == True:
                subset = self.feature_extraction(X_train, y_train, subset, features_len=features_len)

            if pca_reduction == True:
                pca = PCA(n_components=features_len)
                X_train = pca.fit_transform(X_train)
                X_test = pca.fit_transform(X_test)
                # print('PCA score',pca.explained_variance_ratio_,pca.components_)
                # print('PCA zip', zip(subset, pca.explained_variance_ratio_))
                print(pd.DataFrame(X_train).head())
                print(pca.get_params())
            ##todo uncomment to performe feature_extraction


            # subset = ['EURNZDKALMAN', "EURNZDBOLLINGER20", 'EURAUDKALMAN']
            print('subset', subset)
            # Create the (parametrised) models
            print("Hit Rates/Confusion Matrices:\n")
            lr = LogisticRegression()
            qda = QDA()
            lsvc = LinearSVC()

            svc = SVC( C=1000000.0, cache_size=200, class_weight=None,
                          coef0=0.0, degree=3, gamma=0.0001, kernel='rbf',
            max_iter = -1, probability = False, random_state = None,
                                                               shrinking = True, tol = 0.001, verbose = False)

            rfc = RandomForestClassifier( n_estimators=1000, criterion='gini',
            max_depth = None, min_samples_split = 2,
                                                  min_samples_leaf = 1, max_features ='auto',
            bootstrap = True, oob_score = False, n_jobs = 1,
                                                          random_state = None, verbose = 0)

            vc = VotingClassifier(estimators=[('lr', lr), ('rfc', rfc), ('svc', svc), ('qda', qda), ('lsvc', lsvc)], voting='hard')
            adaboost = AdaBoostClassifier()
            gaussiannb = GaussianNB()
            multinb = MultinomialNB()
            bagging = BaggingClassifier(svc)
            models = [
                # ("Logistic Regression (LR)", lr),
                      # ("Linear Discriminant Analyser (LDA)", LDA()),
                      # ("Quadratic Discriminant Analyser (QDA)", qda),
                      # ("Linear Support Vector Classifier (LSVC)", lsvc),
                      # ("Radial Support Vector Machine (RSVM)", svc),
                      # ("Voting Classifier", vc),
                      # ("AdaBoost Classifier", adaboost),
                      ("Bagging Classifier", bagging),
                      # ("Gaussian Naive Bayes", gaussiannb),
                      # ("Multinomial Naive Bayes", multinb),
            # ("Random Forest Classifier(RF)", rfc )
            ]
            df = {}
            result = {}
            # print(X_train)
            scores[str(subset)] = {}
            self.cpu = []
            self.cpu_times = []
            self.wall_time = []
            self.process_time = []
            self.memory = []
            for m in models:
                t0 = time.clock()
                t1 = time.time()
                self.cpu.append((str(m[0]),psutil.cpu_percent(interval=None)))
                self.cpu_times.append((str(m[0]),psutil.cpu_times()))
                self.memory.append((str(m[0]),psutil.virtual_memory()))
                # print('INDEX', np.isnan(X_train['SMA20'].values.sum()))

                m[1].fit(X_train, y_train['Direction'])

                # Make an array of predictions on the test set
                pred = m[1].predict(X_test)

                # fig, ax = plt.subplots()
                # ax.scatter(y_test['Direction'], pred)
                # ax.plot([y_test['Direction'].min(), y_test['Direction'].max()], [y_test['Direction'].min(), y_test['Direction'].max()], 'k--', lw=4)
                # ax.set_xlabel('Measured')
                # ax.set_ylabel('Predicted')
                # plt.show()

                # Output the hit-rate and the confusion matrix for each model
                scores[str(subset)][str(m[0])] = m[1].score(X_test, y_test['Direction'])
                df[str(m[0])] = {}
                df[str(m[0])]['conf_matrix'] = str(confusion_matrix(pred, y_test['Direction']))
                df[str(m[0])]['pred_score'] =  m[1].score(X_test, y_test['Direction'])
                df[str(m[0])]['recall'] = recall_score(pred, y_test['Direction'])

                # TODO ADD CORRELATIONS and RMSE
                print("%s:\nPrediction score: %0.3f" % (m[0], df[str(m[0])]['pred_score']))
                print("Recall Score: %s\n" % df[str(m[0])]['recall'] )
                print("Confusion Matrix\n%s\n\n" % confusion_matrix(pred, y_test['Direction']))

                print(str(m[0])," # forecast # -", time.clock() - t0, "seconds process time")
                print(str(m[0])," # forecast # -", time.time() - t1, "seconds wall time\n")
                self.wall_time.append((str(m[0]), time.time() - t1))
                self.process_time.append((str(m[0]), time.clock() - t0))
            if store == True:

                result['models'] = df
                result['s_file'] = s_file
                result['subset'] = subset
                result['lags'] = lags
                result['rfe'] = rfe
                result['pca'] = pca_reduction
                result['features_len'] = features_len
                # result['pca_variance_ratio'] = str(zip(subset, pca.explained_variance_ratio_))
                # result['pca_components'] = str(zip(subset, pca.components_))
                result['symbol_list'] = symbol_list
                result['execution_time'] = time.time() - t1
                client = MongoClient('localhost', 27017)
                results = client['results']
                mlresults = results.ml
                post_id = mlresults.insert_one(result).inserted_id
                print('id inserido', post_id)
            cputime = {}
            cputime['process_time'] = self.process_time
            cputime['wall_time'] = self.wall_time
            cputime['cpu'] = self.cpu
            cputime['cputimes'] = self.cpu_times
        with open('forecasting_performance.json', 'w') as outfile:
            json.dump(cputime, outfile)


        # bestscore = 0
        # t = []
        # for subset in scores:
        #     # print('subset',subset, scores.keys(), scores.values())
        #     bestscores['len_scores'] = len(scores)
        #     bestscores[subset] = {}
        #     bestscores[subset] = {'model' : max(scores[subset], key=scores[subset].get), 'value' : scores[subset][str(max(scores[subset], key=scores[subset].get))]}
        #     t.append((subset, max(scores[subset], key=scores[subset].get), scores[subset][str(max(scores[subset], key=scores[subset].get))]))
        #     if bestscore < scores[subset][str(max(scores[subset], key=scores[subset].get))]:
        #         bestscore = scores[subset][str(max(scores[subset], key=scores[subset].get))]
        #         bestscores['topscore'] = {'topscore' : bestscore, 'subset' : subset, 'model' : max(scores[subset], key=scores[subset].get)}
        # # bestscores['topscores'] = sorted(scores.items(), key=operator.itemgetter(1)(1))[:2]
        # bestscores['topscores'] = sorted(t, key=lambda tup: tup[2], reverse=True)[:5]
        #
        # with open('forecast_scores.json', 'w') as outfile:
        #     json.dump(bestscores, outfile)



forecast = Forecast()
# files = ['_M1_2012_2013','_M1_2013_2014','_M1_2014_2015','_M1_2015_2016']
# files = ['_H1_2012','_H1_2013','_H1_2014','_H1_2015','_H1_2016']
files = ['_m1_2012']
symbol_list = ['eur/USD']
for s_file in files:
    print('s_file',s_file)
    forecast.runForecast(lags=5, features_len=None, s_file=s_file, symbol_list=symbol_list, pca_reduction=False, store=True, rfe=False)
