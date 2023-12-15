import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

from imblearn.combine import SMOTETomek

#Read in the data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

train = train.drop(['id'], axis=1)
test = test.drop(['id'], axis=1)

target = train['Status']
training = train.drop(['Status'], axis=1)

target.hist()
plt.show()


#print(train.shape, train.head())
#print(test.shape, test.head())

#print(train.columns, test.columns)

#Convert all categorical variables to numerical variables
cat_columns = train.select_dtypes(['object']).columns
train[cat_columns] = train[cat_columns].apply(lambda x: pd.factorize(x)[0])

cat_columns = test.select_dtypes(['object']).columns
test[cat_columns] = test[cat_columns].apply(lambda x: pd.factorize(x)[0])

#print(train.shape, train.head())
#print(test.shape, test.head())

#print(train.columns, test.columns)

print(train.describe())
print(test.describe())

print(train.info())
print(test.info())

print(train.isnull().sum())
print(test.isnull().sum())

#Imbalanced class
#https://www.kaggle.com/rafjaa/resampling-strategies-for-imbalanced-datasets


#Feature selection
import seaborn as sns

correlation_matrix = train.corr()
#correlation_matrix.style.background_gradient(cmap='coolwarm').set_precision(2)

plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.show()


