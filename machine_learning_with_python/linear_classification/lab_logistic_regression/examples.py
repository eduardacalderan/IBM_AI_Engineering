import pandas as pd
import pylab as pl
import numpy as np
import scipy.optimize as opt
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, jaccard_score, classification_report, log_loss
import itertools

# === Logistic Regression === 

##############################################################
# Logistic Regression is a variation of Linear Regression, used when the observed dependent variable, <b>y</b>, is categorical. It produces a formula that predicts the probability of the class label as a function of the independent variables.
##############################################################


##############################################################
# == Customer churn with Logistic Regression ==
# A telecommunications company is concerned about the number of customers leaving their land-line business for cable competitors. They need to understand who is leaving. Imagine that you are an analyst at this company and you have to find out who is leaving and why.
##############################################################


##############################################################
# Load the Telco Churn Data
churn_df = pd.read_csv(r'machine_learning_with_python\linear_classification\lab_logistic_regression\ChurnData.csv')
churn_df.head()
##############################################################


##############################################################
# Data pre-processing and selection
churn_df = churn_df[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip',   'callcard', 'wireless','churn']]
churn_df['churn'] = churn_df['churn'].astype('int') # The target data type must be an integer, as it is a requirement by the SkitLearn algorithm

# -> Count the total rows and columns are in this dataset, and verify the name of the columns 
total_rows_churn_df = churn_df.shape[0]
total_columns_churn_df = churn_df.shape[1]
columns_name_churn_df = churn_df.columns

print(f'Total rows: {total_rows_churn_df}\n')
print(f'Total columns: {total_columns_churn_df}\n')
print(f'Columns name: {columns_name_churn_df}\n')

print(churn_df.shape)

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
print(f'Train set: {X_train.shape}, {y_train.shape}')
print(f'Test set: {X_test.shape}, {y_test.shape}')
##############################################################


##############################################################
# Modeling (Logistic Regression with Scikit-learn)
# LogisticRegression from the Scikit-Learn package implements logistic regression and can use different numerical optimizers to find parameters, including ‘newton-cg’, ‘lbfgs’, ‘liblinear’, ‘sag’, ‘saga’ solvers.
# The version of Logistic Regression in Scikit-learn, support regularization. Regularization is a technique used to solve the overfitting problem of machine learning models. C parameter indicates inverse of regularization strength which must be a positive float. Smaller values specify stronger regularization.

LR = LogisticRegression(C=0.01, solver='liblinear').fit(X_train, y_train)
LR

# -> Predict using our test set 
yhat = LR.predict(X_test)
yhat

yhat_prob = LR.predict_proba(X_test) # predict_proba returns estimates for all classes, ordered by the label of classes. So, the first column is the probability of class 0, P(Y=0|X), and second column is probability of class 1, P(Y=1|X)
yhat_prob
##############################################################


##############################################################
# Evaluation
# -> Jaccard index 
# Using jaccard index for accuracy evaluation: we can define jaccard as the size of the intersection divided by the size of the union of the two label sets. If the entire set of predicted labels for a sample strictly matches with the true set of labels, then the subset accuracy is 1.0; otherwise it is 0.0.
jaccard_score(y_test, yhat, pos_label=0)

# -> Confusion matrix
# That is Another way of looking at the accuracy of the classifier.

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
  """
  This function prints and plots the confusion matrix.
  Normalization can be applied by setting `normalize=True`.
  """
  if normalize:
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print('Normalized confusion matrix')
  else:
    print('Confusion matrix, without normalization')

  print(cm)

  plt.imshow(cm, interpolation='nearest', cmap=cmap)
  plt.title(title)
  plt.colorbar()
  trick_marks = np.arange(len(classes))
  plt.xticks(trick_marks, classes, rotation=45)
  plt.yticks(trick_marks, classes)

  fmt = '2.f' if normalize else 'd'
  thresh = cm.max() / 2
  for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, format(cm[i, j], fmt),
             horizontalalignment="center",
             color="white" if cm[i, j] > thresh else "black")
    
  plt.tight_layout()
  plt.ylabel('True label')
  plt.xlabel('Predicted label')

print(confusion_matrix(y_test, yhat, labels=[1,0]))

# -> Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, yhat, labels=[1,0])
np.set_printoptions(precision=2)

#-> Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['churn=1','churn=0'],normalize=False,  title='Confusion matrix')

print(classification_report(y_test, yhat))
##############################################################


##############################################################
# log loss
# In logistic regression, the output can be the probability of customer churn is yes (or equals to 1). This probability is a value between 0 and 1. Log loss( Logarithmic loss) measures the performance of a classifier where the predicted output is a probability value between 0 and 1.
log_loss(y_test, yhat_prob)
##############################################################