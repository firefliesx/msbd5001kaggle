from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

from sklearn.tree import DecisionTreeRegressor
from sklearn import preprocessing
from keras.models import Sequential
from keras.layers import Bidirectional,LSTM
from keras.layers.core import Dense, Activation, Dropout

#data preprocessing
train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")
sample_submission = pd.read_csv("sampleSubmission.csv")
train_data.dropna(axis=0,how='any',inplace=True)

# def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
#     n_vars = 1 if type(data) is list else data.shape[1]
#     #df = DataFrame(data)
#     df = data
#     cols, names = [], []
#     for i in range(n_in, 0, -1):
#         cols.append(df.shift(i))
#         names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
#     for i in range(0, n_out):
#         cols.append(df.shift(-i))
#         if i == 0:
#             names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
#         else:
#             names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
#     agg = concat(cols, axis=1)
#     agg.columns = names
#     if dropnan:
#         agg.dropna(inplace=True)
#     return agg
# print(series_to_supervised(train_data).head())

def feature_transfer(data):
	data['day'] = data['date'].apply(lambda x: float(x.split('/')[0]))
	data['month'] = data['date'].apply(lambda x: float(x.split('/')[1]))
	data['year'] = data['date'].apply(lambda x: float(x.split('/')[2].split(' ')[0]))
	data['hour'] = data['date'].apply(lambda x: float(x.split('/')[2].split(' ')[1].split(':')[0]))

feature_transfer(train_data)
feature_transfer(test_data)

y_train = np.array(train_data['speed'])
train_data = train_data.drop(['date','id','speed'],axis=1)
test_data = test_data.drop(['date','id'],axis=1)
X_train,X_test = np.array(train_data),np.array(test_data)
X_train = preprocessing.scale(X_train,axis=1)
X_test = preprocessing.scale(X_test,axis=1)
#X_train,X_test = X_train.reshape(X_train.shape[0],1,X_train.shape[1]),X_test.reshape(X_test.shape[0],1,X_test.shape[1])



#model
# model = Sequential()
# model.add(LSTM(input_shape=(None,4),units=50,return_sequences=True))
# model.add(Dropout(0.2))
# model.add(LSTM(input_shape=(None,50),units=50,return_sequences=True))
# model.add(Dropout(0.2))
# model.add(LSTM(input_shape=(None,50),units=50,return_sequences=True))
# model.add(Dropout(0.2))
# model.add(LSTM(50,return_sequences=False))
# model.add(Dropout(0.2))
# model.add(Dense(units=1))
# model.add(Activation("linear"))
# start = time.time()
# model.compile(loss="mse", optimizer="rmsprop")
# print("Time : ", time.time() - start)
# #print(model.layers)

# model.fit(X_train,y_train,batch_size=50,epochs=10,validation_split=0.05)

model = ElasticNet(l1_ratio=0.5,normalize=True,max_iter=15000)
model.set_params(alpha=0.001)
model.fit(X_train,y_train)

y_test = model.predict(X_test)
prediction_result = y_test
#prediction_result = []
# for i in range(len(y_test)):
# 	prediction_result.append(y_test[i][0])
speed_id = [x for x in range(len(y_test))]
#print(type(y_test))
result = pd.DataFrame({'id':speed_id,'speed':prediction_result})
result.to_csv('submission.csv',index=False)
print(y_test)
