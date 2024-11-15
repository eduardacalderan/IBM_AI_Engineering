# Try to use a multiple linear regression with the same dataset, but this time use FUELCONSUMPTION_CITY and FUELCONSUMPTION_HWY instead of FUELCONSUMPTION_COMB. Does it result in better accuracy? 

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import linear_model

# Reading the Data
df = pd.read_csv(r"machine_learning_with_python\regression\lab_multiple_linear_regression\FuelConsumptionCo2.csv")
df.head()
cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_CITY','FUELCONSUMPTION_HWY','FUELCONSUMPTION_COMB','CO2EMISSIONS']]

# Splitting the dataset 
msk = np.random.rand(len(df)) < 0.8 
train = cdf[msk] 
test = cdf[~msk]

regr = linear_model.LinearRegression()
x = np.asanyarray(train[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_CITY','FUELCONSUMPTION_HWY']])
y = np.asanyarray(train[['CO2EMISSIONS']])
regr.fit(x, y)

# Predicting
y_hat = regr.predict(test[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_CITY','FUELCONSUMPTION_HWY']])
x = np.asanyarray(test[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_CITY','FUELCONSUMPTION_HWY']])
y = np.asanyarray(test[['CO2EMISSIONS']])

print("Residual sum of squares : %2.f" % np.mean((y_hat - y)**2))
print('Variance score: %.2f' % regr.score(x, y))