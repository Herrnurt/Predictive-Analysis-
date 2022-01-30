# -*- coding: utf-8 -*-
"""
Created on Mon Nov  8 11:36:58 2021

@author: Agboola Temiremi 
"""


## Classification & Neural Networks

## Prepare Data

# Load libraries
import numpy as np                             # linear algebra
import pandas as pd                            # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt                # MATLAB-like way of plotting
from pandas import set_option

# Load dataset
nba_rookie_data = pd.read_csv('nba_rookie_data.csv')


## Summarize Data

# display dataset in rows and columns
display(nba_rookie_data.head())

# display number of rows and columns
nba_rookie_data.shape

# list the column names
nba_rookie_data.columns



## Descriptive statistics

# descriptions
set_option('precision', 1)
display(nba_rookie_data.describe())


## Data Anlaysis

# correlation
set_option('precision', 3)
display(nba_rookie_data.corr(method='pearson'))

# Considering attributes that are strongly correlated to price
# Using values not less that 0.25 to select the attributes that are strongly correlated to price ,
# Attributes that are strongly correlated with price are :
# Games Played,Minutes Played,Points Per Game,Field Goals Made,Field Goal Attempts
# Free Throw Made,Free Throw Attempts,Offensive Rebounds,Defensive Rebounds,Rebounds,Turnovers

# drop unwanted features
nba_rookie_data_df=nba_rookie_data.drop(['Name','Field Goal Percent','3 Point Made','3 Point Attempt','3 Point Percent','Free Throw Percent','Assists','Steals','Blocks'],axis=1)
#nba_rookie_data_df.head()

# give the first ten faetures toXand the last to y
data =nba_rookie_data_df.values
X= data[:,0:11]
y =data[:,11]

## Logistic Regression Model

# sklearn package for machine learning in python:
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from matplotlib import pyplot

# apply the Logistic regresiion Algorithm to X data
# and fit the data
logre = LogisticRegression()
logre.fit(X, y)


# using arbitary numbers for prediction
test_point = [[44.6,10.3,5.2,3.9,6.7,2.9,2.3,0.9,1.7,5.2,1.1]]
y_pred = logre.predict(np.array(test_point))
print('Prediction:', y_pred)

# print the accuracy score
print('Our Accuracy is %.2f' % logre.score(X, y))

# printNumber of mislabeled points out of a total 
print('Number of mislabeled points out of a total %d points : %d'% (X.shape[0], (y != logre.predict(X)).sum()))




## Inclusion of many features

# remove unwanted features
nba_rookie_data=nba_rookie_data.drop(columns=['3 Point Percent','Name'])

#nba_rookie_data.head()
# give first 9 columns to X and the last to y
X_new= nba_rookie_data.values[:,0:18]
y_new =nba_rookie_data.values[:,18]

# apply the Logistic regresiion Algorithm to X data
# and fit the data
logre = LogisticRegression()
logre.fit(X_new, y_new)

# using Arbitary numbers for prediction
test_point = [[74.4,21.6,8.8,4.6,7.8,46.1,0.6,0.9,3.2,3.8,88.3,3.1,4.3,5.3,3.5,0.9,0.6,3.2]]
y_new_pred = logre.predict(np.array(test_point))
print('Prediction:', y_new_pred)

# print the accuracy score
print('Our Accuracy is %.2f' % logre.score(X_new, y_new))

# print number of mislabeled points out of a total 
print('Number of mislabeled points out of a total %d points : %d'% (X_new.shape[0], (y_new != logre.predict(X_new)).sum()))



## Gaussian Naive Bayes Model

# sklearn package for machine learning in python:
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import make_classification
from sklearn.preprocessing import LabelEncoder

# apply the Gaussian Naive Bayes  Algorithm to X data
# and fit the data
gnb = GaussianNB()
gnb.fit(X, y)

# using Arbitary numbers for prediction
print('Predict a value %d:' % gnb.predict([[44.6,10.3,5.2,3.9,6.7,2.9,2.3,0.9,1.7,5.2,1.1]]))

# print the accuracy score
print('Our Accuracy is %.2f' % gnb.score(X, y))

# print number of mislabeled points out of a total 
print('Number of mislabeled points out of a total %d points : %d'% (X.shape[0], (y != gnb.predict(X)).sum()))


## With Many Features


# apply the Gaussian Naive Bayes  Algorithm to X data
# and fit the data
gnb = GaussianNB()
gnb.fit(X_new, y_new)

# using Arbitary numbers for prediction
print('Predict a value %d:' % gnb.predict([[74.4,21.6,8.8,4.6,7.8,46.1,0.6,0.9,3.2,3.8,88.3,3.1,4.3,5.3,3.5,0.9,0.6,3.2]]))

# print the accuracy score
print('Our Accuracy is %.2f' % gnb.score(X_new, y_new))

# print number of mislabeled points out of a total 
print('Number of mislabeled points out of a total %d points : %d'% (X_new.shape[0], (y_new != gnb.predict(X_new)).sum()))


## Neural Networks Model

# sklearn package for machine learning in python:
from sklearn.neural_network import MLPClassifier

# apply the Neural Networks Algorithm to X data
# and fit the data
mlp = MLPClassifier(hidden_layer_sizes=(), activation="logistic",random_state=0, max_iter=2000)
mlp.fit(X, y)

# using Arbitary numbers for prediction
print('Prediction:', mlp.predict([[44.6,10.3,5.2,3.9,6.7,2.9,2.3,0.9,1.7,5.2,1.1]]))

# print the accuracy score
print('Our Accuracy is %.2f' % mlp.score(X, y))

# print number of mislabeled points out of a total 
print('Number of mislabeled points out of a total %d points : %d'% (X.shape[0], (y != mlp.predict(X)).sum()))


## With many features

# apply the Neural Networks Algorithm to X data
# and fit the data
mlp = MLPClassifier(hidden_layer_sizes=(), activation="logistic",random_state=0, max_iter=2000)
mlp.fit(X_new, y_new)

# using Arbitary numbers for prediction
print('Prediction:', mlp.predict([[74.4,21.6,8.8,4.6,7.8,46.1,0.6,0.9,3.2,3.8,88.3,3.1,4.3,5.3,3.5,0.9,0.6,3.2]]))

# print the accuracy score
print('Our Accuracy is %.2f' % mlp.score(X_new, y_new))

# print number of mislabeled points out of a total 
print('Number of mislabeled points out of a total %d points : %d'% (X_new.shape[0], (y_new != mlp.predict(X_new)).sum()))



## A way of visualising the Datasets
## Visualising the Logistic Regression

import seaborn as sns
%matplotlib inline 

width = 20
height = 18
plt.figure(figsize=(width, height))
sns.regplot(x="Games Played", y="TARGET_5Yrs", data=nba_rookie_data )
plt.ylim(0,)
