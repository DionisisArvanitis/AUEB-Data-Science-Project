import pandas as pd
import tensorflow 
from tensorflow import keras
from keras import optimizers
from keras import regularizers
from keras.models import Sequential
from keras.models import Model
from keras.layers import Dense
from keras.layers import Dropout
from keras.callbacks import ModelCheckpoint


data_test = 'test.csv'
data_train = 'train.csv'

df_train = pd.read_csv(data_train)
df_test = pd.read_csv(data_test)

season_train = pd.get_dummies(df_train.season, prefix ='season')
mnth_train = pd.get_dummies(df_train.mnth, prefix ='mnth')
weekday_train = pd.get_dummies(df_train.weekday, prefix ='weekday')
hr_train = pd.get_dummies(df_train.hr, prefix ='hr')
yr_train = pd.get_dummies(df_train.yr, prefix ='yr')
weathersit_train = pd.get_dummies(df_train.weathersit, prefix ='weathersit')
workingday_train = pd.get_dummies(df_train.workingday, prefix ='workingday')
holiday_train = pd.get_dummies(df_train.holiday, prefix ='holiday')

season_test = pd.get_dummies(df_test.season, prefix ='season')
mnth_test = pd.get_dummies(df_test.mnth, prefix ='mnth')
weekday_test = pd.get_dummies(df_test.weekday, prefix ='weekday')
hr_test = pd.get_dummies(df_test.hr, prefix ='hr')
yr_test = pd.get_dummies(df_test.yr, prefix ='yr')
weathersit_test = pd.get_dummies(df_test.weathersit, prefix ='weathersit')
workingday_test = pd.get_dummies(df_test.workingday, prefix ='workingday')
holiday_test = pd.get_dummies(df_test.holiday, prefix ='holiday')

atmosphere_train = df_train[['temp','atemp','hum','windspeed']]
atmosphere_test = df_test[['temp','atemp','hum','windspeed']]

df_test['weathersit_4']=0

missing_weathersit = df_test[['weathersit_4']]


ohe_test = pd.concat([hr_test,weekday_test,mnth_test,yr_test,season_test,holiday_test,workingday_test,weathersit_test,missing_weathersit,atmosphere_test], axis=1)
ohe_train = pd.concat([hr_train,weekday_train,mnth_train,yr_train,season_train,holiday_train,workingday_train,weathersit_train,atmosphere_train], axis=1)

x = ohe_train
y = df_train[['cnt']]


model = Sequential()
model.add(Dense(228,input_dim=61,activity_regularizer=regularizers.l1(0.001),activation='tanh'))
model.add(Dropout(0.20))
model.add(Dense(228,activity_regularizer=regularizers.l1(0.001),activation='relu'))
model.add(Dropout(0.10))
model.add(Dense(228,activity_regularizer=regularizers.l1(0.001),activation='relu'))
model.add(Dropout(0.10))
model.add(Dense(228,activity_regularizer=regularizers.l1(0.001),activation='relu'))
model.add(Dense(1,activity_regularizer=regularizers.l1(0.001), activation='relu'))


adam = optimizers.adam(learning_rate=0.002)

model.compile(loss='mean_squared_error', optimizer=adam, metrics=['accuracy'])

filepath="weights.best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

model.fit(x, y, validation_split=0.20, epochs=100, batch_size=50, callbacks=callbacks_list, verbose=0)

model.load_weights("weights.best.hdf5")

prediction = model.predict(ohe_test)

submission = pd.DataFrame()
submission['Id'] = range(prediction.shape[0])
submission['Predicted'] = prediction

submission.to_csv("pred.csv", index=False)
