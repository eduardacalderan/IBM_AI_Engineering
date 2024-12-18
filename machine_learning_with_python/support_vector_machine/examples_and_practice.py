"""
Using SVM (Support Vector Machines) to build and train a model using human cell records, and classify cells to whether the samples are benign or malignant.

SVM works by mapping data to a high-dimensional feature space so that data points can be categorized, even when the data are not otherwise linearly separable. A separator between the categories is found, then the data is transformed in such a way that the separator could be drawn as a hyperplane. Following this, characteristics of new data can be used to predict the group to which a new record should belong.
"""
import pandas as pd
import pylab as pl
import numpy as np
import scipy.optimize as opt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import jaccard_score
import itertools
import matplotlib.pyplot as plt

################################################################
# -> Load the Cancer data 
cell_df = pd.read_csv("machine_learning_with_python\support_vector_machine\cell_samples.csv")
cell_df.head()

# The ID field contains the patient identifiers. The characteristics of the cell samples from each patient are contained in fields Clump to Mit. The values are graded from 1 to 10, with 1 being the closest to benign.
# The Class field contains the diagnosis, as confirmed by separate medical procedures, as to whether the samples are benign (value = 2) or malignant (value = 4).

# Distribution of the classes based on Clump thickness and Uniformity of cell size:

ax = cell_df[cell_df['Class'] == 4][0:50].plot(kind='scatter', x='Clump', y='UnifSize', color='DarkBlue', label='malignant');
cell_df[cell_df['Class'] == 2][0:50].plot(kind='scatter', x='Clump', y='UnifSize', color='Yellow', label='benign', ax=ax);
plt.show()
################################################################

################################################################
# -> Data pre-processing and selection

# BareNuc column includes some values that aren't numerical. We will drop those rows.
cell_df = cell_df[pd.to_numeric(cell_df['BareNuc'], errors='coerce').notnull()]
cell_df['BareNuc'] = cell_df['BareNuc'].astype('int')
cell_df.dtypes

feature_df = cell_df[['Clump', 'UnifSize', 'UnifShape', 'MargAdh', 'SingEpiSize', 'BareNuc', 'BlandChrom', 'NormNucl', 'Mit']]
X = np.asarray(feature_df)
X[0:5]

# We need the model to predict the value of Class (that is, benign (=2) or malignant (=4)).
y = np.asarray(cell_df['Class'])
y [0:5]
################################################################

################################################################
# -> Train/Test dataset

# Splitting the dataset into train and test set
X_train, X_test, y_train, y_test, = train_test_split(X, y, test_size=0.2, random_state=4)
print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)################################################################

################################################################
# -> Modeling (SVM with Scikit-learn)
# The SVM algorithm offers a choice of kernel functions for performing its processing. Basically, mapping data into a higher dimensional space is called kernelling. The mathematical function used for the transformation is known as the kernel function, and can be of different types, such as:

# 1.Linear
# 2.Polynomial
# 3.Radial basis function (RBF)
# 4.Sigmoid

# Each of these functions has its characteristics, its pros and cons, and its equation, but as there's no easy way of knowing which function performs best with any given dataset. We usually choose different functions in turn and compare the results. 
# Using the default, RBF (Radial Basis Function):
clf_rbf = svm.SVC(kernel='rbf')
clf_rbf.fit(X_train, y_train)

# After being fitted, the model can then be use to predict new values
yhat = clf_rbf.predict(X_test)
yhat [0:5]
################################################################

################################################################
# -> Evaluation
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
  
# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, yhat, labels=[2,4])
np.set_printoptions(precision=2)

print (classification_report(y_test, yhat))

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['Benign(2)','Malignant(4)'],normalize= False,  title='Confusion matrix')

# Using the f1_score from sklearn library
f1_score(y_test, yhat, average='weighted')

# Using jaccard_score to evaluate the accuracy of the model
jaccard_score(y_test, yhat,pos_label=2)
################################################################

################################################################
"""PRACTICING:
Can you rebuild the model, but this time with a __linear__ kernel? You can use __kernel='linear'__ option, when you define the svm. How the accuracy changes with the new kernel function?
"""

# Using the linear kernel
clf_linear = svm.SVC(kernel='linear')

# Fitting the model
clf_linear.fit(X_train, y_train)

# Predicting the values
yhat_linear = clf_linear.predict(X_test)

# Using the f1_score from sklearn library
accuracy_clf_f1_score = f1_score(y_test, yhat_linear, average='weighted')

# Using jaccard_score to evaluate the accuracy of the model
accuracy_clf_jaccard_score = jaccard_score(y_test, yhat_linear,pos_label=2)

print("Avg F1-score: %.4f" % accuracy_clf_f1_score)
print("Jaccard score: %.4f" % accuracy_clf_jaccard_score)
################################################################