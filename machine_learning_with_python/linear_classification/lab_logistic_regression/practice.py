# Try to build Logistic Regression model again for the same dataset, but this time, use different __solver__ and __regularization__ values? What is new __logLoss__ value?

import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss


##############################################################
# Load the Telco Churn Data
churn_df = pd.read_csv(r'machine_learning_with_python\linear_classification\lab_logistic_regression\ChurnData.csv')
churn_df.head()
##############################################################


##############################################################
# Data pre-processing and selection
churn_df = churn_df[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip',   'callcard', 'wireless','churn']]
churn_df['churn'] = churn_df['churn'].astype('int') # The target data type must be an integer, as it is a requirement by the SkitLearn algorithm

# -> Define X, and y 
X = np.asarray(churn_df[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip']])
X[0:5]

y = np.asarray(churn_df['churn'])
y[0:5]

# -> Normalize the dataset
X = preprocessing.StandardScaler().fit(X).transform(X)
X[0:5]
##############################################################


##############################################################
# Train/Test dataset
# -> Split our dataframe into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)
##############################################################

# PRACTICE: Try to build Logistic Regression model again for the same dataset, but this time, use different __solver__ and __regularization__ values? What is new __logLoss__ value?

LR = LogisticRegression(C=0.01, solver='newton-cg').fit(X_train, y_train)
LR

yhat_prob = LR.predict_proba(X_test) 

print ("LogLoss: : %.2f" % log_loss(y_test, yhat_prob))
