# -*- coding: utf-8 -*-
"""
Created on Sat Nov  6 10:38:47 2021

@author:  Agboola Temiremi 
"""


## Clustering Task 2

##  Prepare Problem

# Load libraries

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # MATLAB-like way of plotting

# sklearn package for machine learning in python:
from pandas import set_option

# Load dataset
country_data = pd.read_csv('country_data.csv')


## Summarize Data

# display dataset in rows and columns
display(country_data.head())

# display number of rows and columns
country_data.shape

# list the column names
country_data.columns


## Descriptive statistics

# descriptions
set_option('precision', 2)
print(country_data.describe())


## Data Analysis

# correlation
set_option('precision', 3)
print(country_data.corr(method='pearson'))

# Considering attributes that are strongly correlated to price
# using income as the basis and considering values that are strongly correlated to it
# Using values with positive values for clustering
# the attributes to be considred are:
# exports,  health , imports,  income, life_expec, gdpp  

# drop the unwanted features
country_data_df = country_data.drop(['child_mort', 'inflation', 'total_fer','country'],axis=1)

# print the first five rows with corresponing columns
country_data_df.head()



## MeanShift Algorithm

# give the 6 columns to X
X= country_data_df.values[:,0:6]

# sklearn package for machine learning in python:
from sklearn.cluster import MeanShift
from sklearn.cluster import KMeans
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D

from sklearn.cluster import  estimate_bandwidth
from mpl_toolkits.mplot3d import Axes3D
bandwidth = estimate_bandwidth(X, quantile=0.2, n_samples=500)

ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
ms.fit(X)
labels = ms.labels_
cluster_centers = ms.cluster_centers_
print(cluster_centers)
n_clusters_ = len(np.unique(labels))
print(n_clusters_)


from sklearn.cluster import  estimate_bandwidth
from mpl_toolkits.mplot3d import Axes3D
bandwidth = estimate_bandwidth(X, quantile=0.2, n_samples=500)

msc_fig = plt.figure(figsize=(12, 10))

ax = msc_fig.add_subplot(111, projection ='3d')

ax.scatter(X[:, 0], X[:, 1], X[:, 2], marker ='o',color ='blue')

ax.scatter(cluster_centers[:, 0], cluster_centers[:, 1],
           cluster_centers[:, 2], marker ='o', color ='green',
          s = 300, linewidth = 5, zorder = 10)
 
plt.show()


## Inclusion of many features

# remove the feature not needed
country_data= country_data.drop(['country'],axis=1)

# give the 9 columns to X
X_new= country_data_df.values[:,0:9]


# apply the Meanshift Algorithm
ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
ms.fit(X_new)
labels = ms.labels_
cluster_centers = ms.cluster_centers_
print(cluster_centers)
n_clusters_ = len(np.unique(labels))
print(n_clusters_)

## A way of visualising the Datasets
# plot the figure for the datasets clustering
msc_fig2 = plt.figure(figsize=(12, 10))

ax = msc_fig2.add_subplot(111, projection ='3d')

ax.scatter(X[:, 0], X[:, 1], X[:, 2], marker ='o',color ='red')

ax.scatter(cluster_centers[:, 0], cluster_centers[:, 1],
           cluster_centers[:, 2], marker ='o', color ='green',
          s = 300, linewidth = 5, zorder = 10)
 
plt.show()



