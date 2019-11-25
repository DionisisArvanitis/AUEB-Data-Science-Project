# Importing the libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
from scipy import stats
from numpy import median
from sklearn import metrics
from sklearn import model_selection
from sklearn.metrics import mean_squared_log_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

# Reading training data from the local storage
filename = 'train.csv' 
df_train = pd.read_csv(filename)
df_train.head(5)

df_train.shape

df_test = pd.read_csv('test.csv')
df_test.head()

df_test.shape

df_train.isnull().sum()

df_train.rename(columns={'weathersit':'weather',
                     'mnth':'month',
                     'hr':'hour',
                     'yr':'year',
                     'hum': 'humidity',
                     'cnt':'count'},inplace=True)

df_train.head()

df_train.dtypes

df_train['season'] = df_train.season.astype('category')
df_train['year'] = df_train.year.astype('category')
df_train['month'] = df_train.month.astype('category')
df_train['hour'] = df_train.hour.astype('category')
df_train['holiday'] = df_train.holiday.astype('category')
df_train['weekday'] = df_train.weekday.astype('category')
df_train['workingday'] = df_train.workingday.astype('category')
df_train['weather'] = df_train.weather.astype('category')


df_train.dtypes

fig,[ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8] = plt.subplots(nrows=8, figsize=(15,25))
sn.barplot(x = df_train['season'], y = df_train['count'],ax = ax1)
sn.barplot(x = df_train['year'], y = df_train['count'],ax = ax2)
sn.barplot(x = df_train['month'], y = df_train['count'],ax = ax3)
sn.barplot(x = df_train['hour'], y = df_train['count'],ax = ax4)
sn.barplot(x = df_train['holiday'], y = df_train['count'],ax = ax5)
sn.barplot(x = df_train['weekday'], y = df_train['count'],ax = ax6)
sn.barplot(x = df_train['workingday'], y = df_train['count'],ax = ax7)
sn.barplot(x = df_train['weather'], y = df_train['count'],ax = ax8)

# Regression plot is used to verify if a pattern can be observed between `count` and numerical variables

fig,[ax1,ax2,ax3] = plt.subplots(ncols = 3, figsize = (20,8))
plt.rc('xtick', labelsize=10) 
plt.rc('ytick', labelsize=10) 

sn.regplot(x = 'temp', y = 'count',data = df_train, ax = ax1)
ax1.set(title="Relation between temperature and count")
sn.regplot(x = 'humidity', y = 'count',data = df_train, ax = ax2)
ax2.set(title="Relation between humidity and count")
sn.regplot(x = 'windspeed', y = 'count',data = df_train, ax = ax3)
ax3.set(title="Relation between windspeed and count")


data_corr = df_train[['temp', 'atemp', 'humidity', 'windspeed', 'casual', 'registered', 'count']].corr()
mask = np.array(data_corr)
mask[np.tril_indices_from(mask)] = False
fig = plt.subplots(figsize=(15, 10))
sn.heatmap(data_corr, mask=mask, vmax=1, square=True, annot=True, cmap="YlGnBu")

df_train = df_train.drop(['atemp', 'casual', 'registered', 'windspeed'], axis=1)

df_train.head()

# Training and test data is created by splitting the main data. 30% of test data is considered
X = df_train[['temp', 'humidity', 'workingday']]
y = df_train['count']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

X_train.shape

y_train.shape

clf = LinearRegression()
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

True in (y_pred < 0)

for i, y  in enumerate(y_pred):
    if y_pred[i] < 0:
        y_pred[i] = 0

print('RMSLE:', np.sqrt(mean_squared_log_error(y_test, y_pred)))
print('R2:', r2_score(y_test, y_pred))

ax = sn.regplot(y_test, y_pred, x_bins = 200)
ax.set(title = "Comparison between the actual vs predicted values")