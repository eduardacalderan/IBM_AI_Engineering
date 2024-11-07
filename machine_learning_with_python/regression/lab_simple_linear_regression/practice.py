"""
Exercise
Lets see what the evaluation metrics are if we trained a regression model using the FUELCONSUMPTION_COMB feature.
Start by selecting FUELCONSUMPTION_COMB as the train_x data from the train dataframe, then select FUELCONSUMPTION_COMB as the test_x data from the test dataframe
"""

# Importing libs
import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np

from sklearn import linear_model
from sklearn.metrics import r2_score

# Reading the data 
df = pd.read_csv(r"machine_learning_with_python\regression\lab_simple_linear_regression\FuelConsumptionCo2.csv")
df.head()

# Data Exploration
cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
cdf.head(9)

# Creating train and test dataset
## Let's split our dataset into train (80%) and test sets (20%).
msk = np.random.rand(len(df)) < 0.8
train = cdf[msk]
test = cdf[~msk]

# Train a Linear Regression Model using the train_x you created and the train_y as CO2EMISSIONS
regr = linear_model.LinearRegression()
train_x = np.asanyarray(train[['FUELCONSUMPTION_COMB']])
train_y = np.asanyarray(train[['CO2EMISSIONS']])
regr.fit(train_x, train_y)
print('Coefficients: ', regr.coef_)
print('Intercept: ', regr.intercept_)

# Find the predictions using the model's predict function and the test_x data
test_x = np.asanyarray(test[['FUELCONSUMPTION_COMB']])
test_y = np.asanyarray(test[['CO2EMISSIONS']])
predictions = regr.predict(test_x)

# Use the predictions and the test_y data and find the Mean Absolute Error value using the np.absolute and np.mean function like done previously
mean_absolute_error = np.mean(np.absolute(predictions - test_y))

# Find Residual sum of squares (MSE) and R2-score
mse = np.mean((predictions - test_y)**2)
r_two_score = r2_score(test_y, predictions)

print(f"Mean absolute error: {mean_absolute_error}")
print(f"Residual sum of squares (MSE): {mse}")
print(f"R2-score: {r_two_score}")