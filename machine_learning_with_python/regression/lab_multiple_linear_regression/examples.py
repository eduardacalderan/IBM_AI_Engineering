import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import linear_model

### Understanding the Data ###

# FuelConsumptionCo2.csv contains model-specific fuel consumption ratings and estimated carbon dioxide emissions for new light-duty vehicles for retail sale in Canada.

# Reading the Data
df = pd.read_csv(r"machine_learning_with_python\regression\lab_multiple_linear_regression\FuelConsumptionCo2.csv")
df.head()
cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_CITY','FUELCONSUMPTION_HWY','FUELCONSUMPTION_COMB','CO2EMISSIONS']]

# Plotting Emission values with respect to Engine Size
plt.scatter(cdf.ENGINESIZE, cdf.CO2EMISSIONS, color='blue')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show()

# Splitting the dataset into train (80%) and test (20%) sets. 
msk = np.random.rand(len(df)) < 0.8 # Creating a mask to select random rows
train = cdf[msk] 
test = cdf[~msk]

# Train dta distribution
plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS, color='blue')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show()

### Multiple Regression Model ###
# -> When more than one independent variable is present. 
regr = linear_model.LinearRegression()
x = np.asanyarray(train[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']])
y = np.asanyarray(train[['CO2EMISSIONS']])
regr.fit(x, y)
print('Coefficients: ', regr.coef_)

# Ordinary Least Squares (OLS) 
# -> Is a method to estimate the unknown parameters in a linear regression model by minimizing the sum od the squares of the differences between the target dependent variable and those predicted by the linear function.

## Prediction ##
y_hat = regr.predict(test[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']])
x = np.asanyarray(test[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']])
y = np.asanyarray(test[['CO2EMISSIONS']])

print("Mean Squared Error (MSE) : %2.f" % np.mean((y_hat - y)**2))

# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % regr.score(x, y))