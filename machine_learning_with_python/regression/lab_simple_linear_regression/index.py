# Importing libs
import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np

# Reading the data 
df = pd.read_csv("FuelConsumption.csv")
df.head()

# Data Exploration
cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
cdf.head(9)

# Plot each of these features
viz = cdf[['CYLINDERS','ENGINESIZE','CO2EMISSIONS','FUELCONSUMPTION_COMB']]
viz.hist()
plt.show()

# Plot each of these features against the Emission to see how linear their relationship is:

## Plot FUELCONSUMPTION_COMB vs Emission
plt.scatter(cdf.FUELCONSUMPTION_COMB, cdf.CO2EMISSIONS,  color='blue')
plt.xlabel("FUELCONSUMPTION_COMB")
plt.ylabel("Emission")
plt.show()

## Plot Engine size vs Emission
plt.scatter(cdf.ENGINESIZE, cdf.CO2EMISSIONS,  color='blue')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show()