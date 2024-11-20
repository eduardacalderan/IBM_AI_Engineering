# To build the same model of example again, but this time with k=6

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

import pandas as pd

# Reading the data
df = pd.read_csv(r'machine_learning_with_python\classification\lab_knn\teleCust1000t.csv')
print(df.head())

# Defining feature set, X 
X = df[['region','tenure','age','marital','address','income','ed','employ','retire','gender','reside']].values
print('X: ', X[0:5])

# Define y
y = df['custcat'].values  
print('y: ', y[0:5])

# Normalizing Data 
X = preprocessing.StandardScaler().fit(X).transform(X.astype(float))
print('X', X[0:5])

# Train Test SPlit 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=2, random_state=4)
print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)

# Training 
k = 6
neigh = KNeighborsClassifier(n_neighbors=k ).fit(X_train, y_train)
print('neigh: ', neigh)

# Predicting (yhat)
yhat = neigh.predict(X_test)
print('yhat: ', yhat[0:5])

# Accuracy evaluation 
print("Train Set Accuracy: ", metrics.accuracy_score(y_train, neigh.predict(X_train)))
print("Test Set Accuracy: ", metrics.accuracy_score(y_test, yhat))


