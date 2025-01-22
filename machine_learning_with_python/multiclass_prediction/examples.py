import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pandas as pd

from utils import decision_boundary, plot_probability_array

# In ths lab we will use the iris dataset, it consists of three different types of irises’ (Setosa y=0, Versicolour y=1, and Virginica y=2), petal and sepal length, stored in a 150x4 numpy.ndarray.

# The rows being the samples and the columns: Sepal Length, Sepal Width, Petal Length and Petal Width.

# The following plot uses the second two features:

pair = [1, 3] # These integers represent the indices of the features you want to select from the dataset.
iris = datasets.load_iris() # This line loads the Iris dataset using the load_iris function from the datasets module.
X = iris.data[:, pair] # This line selects the columns (features) of the Iris dataset specified by the pair list. iris.data is a 2D array where rows represent samples and columns represent features. [:, pair] means "select all rows and only the columns at indices 1 and 3". The result is stored in the variable X.
y = iris.target # This line assigns the target values (class labels) of the Iris dataset to the variable y. iris.target is a 1D array where each element represents the class label of the corresponding sample.
np.unique(y) # This line uses the unique function from the numpy library to find the unique class labels in y. The result is an array of unique class labels present in the dataset

plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu)
plt.xlabel("sepal width (cm)")
plt.ylabel("petal width")

# Softmax Regression
lr = LogisticRegression(random_state=0).fit(X, y)

probability = lr.predict_proba(X)

plot_probability_array(X,probability)

probability[0, :]

probability[0, :].sum()

# applying the argmax function
np.argmax(probability[0, :])

# applying the argmax function to each sample
softmax_prediction = np.argmax(probability, axis=1)
softmax_prediction

# We can verify that sklearn does this under the hood by comparing it to the output of the method predict
yhat = lr.predict(X)
accuracy_score(yhat, softmax_prediction)

# We can't use Softmax regression for SVMs, Let's explore two methods of Multi-class Classification that we can apply to SVM.