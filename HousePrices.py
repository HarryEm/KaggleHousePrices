import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn import linear_model
from scipy import stats

df_train = pd.read_csv('train.csv')

df_train.shape
df_train.head()
df_train.columns
df_train.dtypes # do the object variables need to be converted to categorical?

df_train['SalePrice']

# The goal is to use the test set variables to predict the value SalesPrice and achieve the lowest RMSE
# compared to the actual prices, which are not observed.

# Detailed description of columns can be found here
# https://storage.googleapis.com/kaggle-competitions-data/kaggle/5407/data_description.txt?GoogleAccessId=web-data@kaggle-161607.iam.gserviceaccount.com&Expires=1519146323&Signature=nbaOkAKyhyfH0eXIZupIMaxmS46cadJQ7CAumW4DR8iWrhugtOhCHu3VOkAe016nJ4Wd0T7%2FRAZqMu3Ee7PDcF4C6G6ng4xPme0wJO1dgurM%2BEW%2BWIFai8JUaSAZnTQhx%2FiPcQhjFGOz04%2BCizyruE4swuJsVMXA33pP0YWfEIU9m4%2FjiAMYKeLlyl%2B%2F3psCI1PekFgpsIxN%2BSksJ%2FDWzG%2BvtcFwbS81WOolqYHjmNU9Z4mXfMJZGqs7g%2Fz5YRaOIj5pkcl4gOthymA0o1AN8pRBl7eOggpe4Lkuuu8Y7STFXPzquq33Unglxt%2B2dePP%2Bi4FXhvIG%2FdraJmtbBiYLg%3D%3D

# What are my intuitions about what the data will look like? It seems obvious to say that the larger,
# more central, more modern the property is, the more it will be worth. An important consideration
# will be to decide how to summarise the dependent variable SalePrice, is there a better method than
# just looking at the mean? I imagine there will be some important decisions around missing data too.

# There are a lot of predictor variables here, which makes me worry about overfitting, and thus I intend
# to use this to measure the effects of regularisation. As I have never fitted a model in Python before,
# my initial approach will be to identify a handful of variables which seem to contain predicative power,
# just drop all the missing values, and fit a simple linear regression.

# "What's the smallest step I can take to make progress"


print((df_train == 0).sum())
print(df_train.isnull().sum())

# .replace(0, numpy.NaN)
#dataset.dropna(inplace=True)
# df1.dropna(how='any')

df_train['MSZoning'] = df_train['MSZoning'].astype('category')

df_train['SalePrice'].hist()
df_train.boxplot(column='SalePrice')
df_train.boxplot(column='SalePrice',by ='MSSubClass')
df_train.boxplot(column='SalePrice',by ='MSZoning')
df_train.boxplot(column='SalePrice',by ='Utilities')
df_train.boxplot(column='SalePrice',by ='Neighborhood')
df_train.boxplot(column='SalePrice',by ='BldgType')
df_train.boxplot(column='SalePrice',by ='OverallQual')
df_train.boxplot(column='SalePrice',by ='OverallCond')
df_train.boxplot(column='SalePrice',by ='Foundation')
df_train.boxplot(column='SalePrice',by ='BedroomAbvGr')

plt.scatter(df_train['1stFlrSF'],df_train['SalePrice'])
plt.xlabel('1st Floor Square Footage')
plt.ylabel('Sale Price')
plt.title('Sale Price vs 1st Floor Sq Footage')
plt.show()


# First attempt at modelling, use entire dataset

split = round(len(df_train)*0.7)

train = df_train[:split]
test = df_train[split:]

regr = linear_model.LinearRegression()

train.loc[:,'SalePrice']
train.drop(['SalePrice'],axis =1)

regr.fit(train.drop(['SalePrice'],axis =1),train.loc[:,'SalePrice'])

# This didn't work, not surprising as I have not managed the data at all.
# Next attempt is to get a model working with a very small subset of features


regr.fit(train.loc[:,'1stFlrSF'],train.loc[:,'SalePrice'])