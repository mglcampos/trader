
from classifier import Classifier, FeatureGenerator
import pandas as pd

ticker = 'EURUSD'
price_data = pd.read_csv('C:\\Users\\utilizador\\Documents\\quant_research\\data\\basic_{}_sample.csv'.format(ticker.lower()))[['Date','Close']]
price_data.columns = ['Date', ticker]

price_data = price_data.drop(['Date'], axis=1)

recent_data = pd.read_csv('C:\\Users\\utilizador\\Documents\\quant_research\\data\\{}15.csv'.format(ticker), names=['Date','Time','Open', 'High', 'Low', 'Close', 'Volume' ])[['Time', 'Close']]
recent_data.columns = ['Time', ticker]
recent_data = recent_data.drop(['Time'], axis=1)
price_data = price_data.append(recent_data[-300:], ignore_index=True)

split = int(len(price_data.index) * 0.85)
train_price_data = price_data[:split]
test_price_data = price_data[split:]

X = train_price_data
cls = Classifier(ticker)
cls.update_sample(X.copy())

cls.train()

h = 0
m = 0
false_up = 0
false_down = 0
pred = 0
previous_state_means = cls.feature_gen.state_means[-1]
previous_price = train_price_data.values[-1]

for row in test_price_data.values:
    if pred > 2000:
        break
    pred +=1
    print("Iter: {}, row: {}".format(pred, [row]))

    X = X.append(pd.DataFrame([row], columns=[ticker]), ignore_index=True)
    X = X.iloc[1:]
    X.index = X.index.values + pred
    cls.update_sample(X.copy())
    x_vec = cls.get_x_vec()
    print("\nState_means change %: {}.".format((cls.feature_gen.state_means[-1]-previous_state_means )/previous_state_means * 100 , cls.feature_gen.price_data['FReturns'].values[-1]*100))
    print("Price change %: {}\n".format(
        (row[0] - previous_price) / previous_price * 100))
    previous_price = row[0]
    previous_state_means = cls.feature_gen.state_means[-1]
    y_pred = cls.predict(x_vec)[0]
    y_label = cls.Y.values[-2]
    print("Label: {}, Prediction: {}".format(y_label, y_pred))
    if y_pred == y_label:
        h +=1
    else:
        if y_label == -1.0:
            false_up += 1
        elif y_label == 1.0:
            false_down += 1
        m +=1
    print("Report: Hits: {}, Missed: {}, False_Up: {}, False_Down: {}.\n\n".format(h, m, false_up, false_down))
    cls.train()

print("Report: Hits: {}, Missed: {}, False_Up: {}, False_Down: {}.".format(h, m, false_up, false_down))


#cls.update_sample(X)
#print(cls.X.head())
#cls.train()
#cls.predict(x_vec)
