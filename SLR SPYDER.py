import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv(r"C:\Users\rosha\OneDrive\Desktop\Gitesh\NIT\TOPICS\ML\Salary_Data.csv")

x = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state=0)


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

y_pred = regressor.predict(x_test)


plt.scatter(x_test, y_test, color = 'red')
plt.plot(x_train, regressor.predict(x_train), color = 'blue')
plt.title('Salary vs Experience(Test set)')
plt.xlabel('Years of Experience(Test set)')
plt.ylabel('Salary')
plt.show()

m = regressor.coef_

c = regressor.intercept_

(m*12) + c
(m*20) + c


bias = regressor.score(x_train, y_train)
bias 


variance = regressor.score(x_test, y_test)
variance 

#stats for slr model

dataset.mean() 
dataset['Salary'].mean()


dataset.median()
dataset['Salary'].median()



dataset['Salary'].mode()


dataset.var()


dataset['Salary'].var()


#Standard deviation

dataset.std()
dataset['Salary'].std()


#Coefficient of Variation(cv)

from scipy.stats import variation
variation(dataset.values)


variation(dataset['Salary'])


#Correlation

dataset.corr()


dataset['Salary'].corr(dataset['YearsExperience'])


#Skewness

dataset.skew()
dataset['Salary'].skew()


#Standard Error

dataset.sem()
dataset['Salary'].sem()


#Z-Score

import scipy.stats as stats
dataset.apply(stats.zscore)


stats.zscore(dataset['Salary'])


#Degree of Freedom


a = dataset.shape[0]
b = dataset.shape[1]

degree_of_freedom = a-b
print(degree_of_freedom)


#SSR (Sum of Sequares Regression)
y_mean = np.mean(y)
SSR = np.sum((y_pred-y_mean)**2)
print(SSR)

#SSE
y = y[0:6]
SSE = np.sum((y-y_pred)**2)
print(SSE)


#SST
mean_total = np.mean(dataset.values)
SST = np.sum((dataset.values-mean_total)**2)
print(SST)


#r2
r_square = 1 - SSR/SST
print(r_square)


import pickle

filename = 'linear_regression_model.pkl'

with open(filename, 'wb') as file:
    pickle.dump(regressor, file)
    
print("Model has been pickled and saved as linear_regression_model.pkl")

import os
os.getcwd()