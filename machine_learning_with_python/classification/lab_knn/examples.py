# === K-Nearest Neighbors === 

##############################################################
# -> KNN is a supervised learning algorithm where the data is trained with data points corresponding to their classification. To predict the class of a given data point, it takes into account the classes of the 'K' nearest data points and chooses the class in which the majority of the 'K' nearest data points belong to as the predicted class.
##############################################################

##############################################################
# Imagine a telecommunications provider has segmented its customer base by service usage patterns, categorizing the customers into four groups. If demographic data can be used to predict group membership, the company can customize offers for individual prospective customers. It is a classification problem. That is, given the dataset, with predefined labels, we need to build a model to be used to predict class of a new or unknown case.
# The example focuses on using demographic data, such as region, age, and marital, to predict usage patterns.
# The target field, called custcat, has four possible values that correspond to the four customer groups, as follows: 
# 1- Basic Service 
# 2- E-Service 
# 3- Plus Service 
# 4- Total Service
# Our objective is to build a classifier, to predict the class of unknown cases. We will use a specific type of classification called K nearest neighbour.
##############################################################

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

import pandas as pd

# Data Visualization and Analysis 
df = pd.read_csv(r'machine_learning_with_python\classification\lab_knn\teleCust1000t.csv')
df.head()
print(df.head())

df['custcat'].value_counts() # -> 266 Basic Service, 217 E-Service, 281 Plus Service, and 236 Total Service
df.hist(column='income', bins=50)
print(df.hist(column='income', bins=50))

# -> Defining feature set, X 
df.columns 
X = df[['region', 'tenure','age', 'marital', 'address', 'income', 'ed', 'employ','retire', 'gender', 'reside']].values
X[0:5]
print(X[0:5])

# -> Defining y 
y = df['custcat'].values
y[0:5]
print(y[0:5])

# Normalizing Data 
X = preprocessing.StandardScaler().fit(X).transform(X.astype(float)) # To use Scikit-Learn, pandas data frame have to be convert to a numpy array
X[0:5]
print(X[0:5])

# -> Train Test Split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)
print('Train set:', X_train.shape, y_train.shape)
print('Test set:', X_test.shape, y_test.shape)

# Classification - K nearest neighbor (knn) 

# -> Training 
k = 4
neigh = KNeighborsClassifier(n_neighbors=k).fit(X_train, y_train)
neigh

# -> Predicting 
yhat = neigh.predict(X_test)
yhat[0:5]
print(yhat[0:5])

# -> Accuracy evaluation 
# In multilabel classification, accuracy classification score is a function that computes subset accuracy. This function is equal to the jaccard_score function. Essentially, it calculates how closely the actual labels and predicted labels are matched in the test set.
print('Train set Accuracy: ', metrics.accuracy_score(y_train, neigh.predict(X_train)))
print('Test set Accuracy: ', metrics.accuracy_score(y_test, yhat))