# -*- coding: utf-8 -*-
"""
Created on Thu Oct 21 18:41:31 2021

@author: kavita.gaikwad
"""

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

dataset = pd.read_csv("Salary_dataset.csv")

## Divide the Dataset into X and y
X= dataset.iloc[:,:-1].values
y = dataset.iloc[:,1].values

## Splitting the data set based on training and test set 
#from sklearn.cross_validation import tran_test_split
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=1/3,random_state=0)

## Implement our classifier based on simple Linear Regression
from sklearn.linear_model import LinearRegression
simpleLinearRegression = LinearRegression()
simpleLinearRegression.fit(X_train,y_train)

y_predict = simpleLinearRegression.predict(X_test)


##y_predict_val = simpleLinearRegression.predict(11)

##implemnt the grapgh 
plt.scatter(X_train,y_train,color='red')
plt.plot(X_train,simpleLinearRegression.predict(X_train))
plt.show()
