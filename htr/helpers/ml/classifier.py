
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.svm import SVC
#from sklearn.pipeline import Pipeline
from sklearn import mixture as mix
import pandas as pd
import datetime as dt
from pykalman import KalmanFilter

class Classifier:

    def __init__(self, ticker):
        self.model = SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0, decision_function_shape=None, degree=3, gamma='auto', kernel='rbf', max_iter=-1, probability=False, random_state=None, shrinking=True, tol=0.001, verbose=False)
        self.feature_gen = FeatureGenerator(ticker)
        #self.X, self.Y = self.feature_gen.get_sample(price_data)

    def predict(self, x_vec):
        return self.model.predict(x_vec)

    def train(self):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.Y, test_size=0.1)
        self.model.fit(X_train, y_train)

    def update_sample(self, price_data):
        self.X, self.Y = self.feature_gen.get_sample(price_data)

    def get_x_vec(self):
        #print(self.X.tail(1))
        return [self.X.values[-1]]

class FeatureGenerator():

    def __init__(self, ticker, n_lags=1, period=14):
        self.period = period
        self.ticker = ticker
        self.n_lags = n_lags

    def get_sample(self, price_data):
        self.price_data = price_data
        self.price_data['Stoch_Osc'] = self.get_stochastic(self.filter_prices())
        self.price_data['FClose'] = pd.DataFrame(self.state_means, index=self.price_data.index, columns=['FClose'])
        #self.price_data['Returns'] =  pd.DataFrame(np.log(self.price_data[self.ticker].values) - np.log(np.roll(self.price_data[self.ticker].values,1)), index=self.price_data.index, columns=['Returns'])
        self.price_data['FReturns'] =  pd.DataFrame(np.log(self.state_means) - np.log(np.roll(self.state_means,1)), index=self.price_data.index, columns=['Returns'])
        #self.price_data['FHurst'] = pd.DataFrame(self.state_means, index = self.price_data.index, columns=['FHurst']).rolling(self.period).apply(self.hurst)
        X, Y = self.generate_regimes(self.generate_lags(self.price_data))
        #print(X.head())
        return X, Y

    def filter_prices(self):
        kf = KalmanFilter(transition_matrices=[1],
                          observation_matrices=[1],
                          initial_state_mean=self.price_data[self.ticker].values[0],
                          initial_state_covariance=1,
                          observation_covariance=1,
                          transition_covariance=.01)

        state_means, _ = kf.filter(self.price_data[self.ticker].values)
        self.state_means = state_means.flatten()
        return state_means

    def hurst(self, p):
        tau = [];
        lagvec = []
        #  Step through the different lags
        for lag in range(2, 20):
            #  produce price difference with lag
            pp = np.subtract(p[lag:], p[:-lag])
            #  Write the different lags into a vector
            lagvec.append(lag)
            #  Calculate the variance of the difference vector
            tau.append(np.sqrt(np.std(pp)))
        #  linear fit to double-log graph (gives power)
        m = np.polyfit(np.log10(lagvec), np.log10(tau), 1)
        # calculate hurst
        hurst = m[0] * 2
        return hurst

    def get_stochastic(self, state_means):
        rolling_min = pd.DataFrame(state_means).rolling(self.period).min()
        rolling_max = pd.DataFrame(state_means).rolling(self.period).max()
        sample = pd.concat([pd.DataFrame(state_means), pd.DataFrame(self.price_data[self.ticker].values), rolling_min, rolling_max],
                           axis=1)
        sample.index = self.price_data.index
        sample.columns = ['FPrice', 'Price', 'Min', 'Max']
        #sample = sample.dropna()
        stoch_osc = (sample.FPrice - sample.Min) / (sample.Max - sample.Min) * 100

        return stoch_osc.values

    def generate_lags(self, X):
        #X = X.drop(['Date'], axis=1)
        lags = 1
        X = self.clean_dataset(X)

        columns = [c for c in X.columns if c not in ['FReturns', 'Returns']]
        for column in columns:
            for lag in range(1, self.n_lags + 1):
                X[column + str(lag)] = X[column].shift(lag)

        X.drop(columns, axis=1, inplace=True)
        X = self.clean_dataset(X)

        return X

    def generate_regimes(self, X):

        ss = preprocessing.StandardScaler()
        split = 2500
        unsup = mix.GaussianMixture(n_components=4, covariance_type="spherical", n_init=100, random_state=42)
        unsup.fit(np.reshape(ss.fit_transform(X[:split]), (-1, X.shape[1])))
        regime = unsup.predict(np.reshape(ss.fit_transform(X[split:]), (-1, X.shape[1])))
        Regimes = pd.DataFrame(regime, columns=['Regime'], index=X[split:].index).join(X[split:], how="inner").assign(
            market_cu_return=X.FReturns[split:].cumsum()).reset_index(drop=False).rename(columns={'index': 'Date'})
        ss1 = preprocessing.StandardScaler()
        columns = Regimes.columns.drop(['Regime', 'Date'])
        Regimes[columns] = ss1.fit_transform(Regimes[columns])
        Y = np.sign(Regimes.FReturns)
        Regimes = Regimes.drop(['FReturns','market_cu_return'], axis=1)
        print(Regimes.tail(1))
        return Regimes, Y

    def clean_dataset(self, df):
        assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
        df.dropna(inplace=True)
        indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
        return df[indices_to_keep].astype(np.float64)