import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

### Understanding the Data ###

# FuelConsumptionCo2.csv contains model-specific fuel consumption ratings and estimated carbon dioxide emissions for new light-duty vehicles for retail sale in Canada.

# Reading the Data
df = pd.read_csv("machine_learning_with_python\regression\lab_multiple_linear_regression\FuelConsumptionCo2.csv")
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
 