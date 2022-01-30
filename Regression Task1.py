# -*- coding: utf-8 -*-
"""
Created on Thur Nov  4 08:57:44 2021

@author: Agboola Temiremi 
"""


## Regression Task 1



##  Prepare Problem

# Load libraries
import numpy as np                # linear algebra
import pandas as pd               # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt   # MATLAB-like way of plotting

# sklearn package for machine learning in python:
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error, r2_score
from pandas import set_option

# Load dataset
house_price = pd.read_csv('houseprice_data.csv')


## Summarize Data

# display dataset in rows and columns
display(house_price.head())

# display number of rows and columns
house_price.shape

# list the column names
house_price.columns


## Descriptive statistics

# descriptions
set_option('precision', 1)
print(house_price.describe())


## Data Analysis

# correlation
set_option('precision', 3)
print(house_price.corr(method='pearson'))

# Considering attributes that are strongly correlated to price
# Using values not less that 0.25 to select the attributes that are strongly correlated to price ,
# Attributes that are strongly correlated with price are :
# bedrooms,bathrooms,sqft_living,floors, waterfront,view,condition,grade,sqft_basement  

# Eliminate some features 
house_price_df=house_price.drop(['sqft_lot','sqft_above','yr_built','yr_renovated','zipcode','lat','long','sqft_living15','sqft_lot15'],axis=1)

# prints the first 5 rows with the corresponding columns
house_price_df.head()

# split the dataset between X and y
# column 1 to column 9 goes to X
# column 0 goes to y,in pandas 0 begins counting 
data = house_price_df.values
X= data[:,1:9]
y =data[:,0]

# split the data into training and test sets:
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 1/3, random_state=0)

# fit the linear least-squres regression line to the training data:
regr = LinearRegression()
regr.fit(X_train, y_train)

# The coefficients
print('Coefficients: ', regr.coef_)

# The intercept
print('Intercept: ', regr.intercept_)

# The mean squared error
print('Mean squared error: %.8f' % mean_squared_error(y_test, regr.predict(X_test)))

# The R^2 value:
print('Coefficient of determination: %.2f' % r2_score(y_test, regr.predict(X_test)))



## Inclusion of many features

# split the dataset between X and y
# column 1 to column 19 goes to X
# column 0 goes to y,in pandas 0 begins counting 
data2 = house_price.values
X_new = data2[:,1:19]
y_new =data2[:,0]

# split the data into training and test sets:
X_new_train, X_new_test, y_new_train, y_new_test = train_test_split(X_new, y_new, test_size= 1/3, random_state=0)

# fit the linear least-squares regression line to the training data:
regr = LinearRegression()
regr.fit(X_new_train, y_new_train)

# The coefficients
print('Coefficients: ', regr.coef_)

# The intercept
print('Intercept: ', regr.intercept_)

# The mean squared error
print('Mean squared error: %.8f' % mean_squared_error(y_new_test, regr.predict(X_new_test)))

# The R^2 value:
print('Coefficient of determination: %.2f' % r2_score(y_new_test, regr.predict(X_new_test)))



## A way of visualising the Datasets

import seaborn as sns
%matplotlib inline 

width = 20
height = 18
plt.figure(figsize=(width, height))
sns.regplot(x="bathrooms", y="price", data=house_price )
plt.ylim(0,)