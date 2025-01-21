import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pandas as pd

# In ths lab we will use the iris dataset, it consists of three different types of irises’ (Setosa y=0, Versicolour y=1, and Virginica y=2), petal and sepal length, stored in a 150x4 numpy.ndarray.

# The rows being the samples and the columns: Sepal Length, Sepal Width, Petal Length and Petal Width.

# The following plot uses the second two features:

pair = [1, 3]
iris = datasets.load_iris()
X = iris.data[:, pair] # we only take the two features
y = iris.target
np.unique(y)

plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu)
plt.xlabel("sepal width (cm)")
plt.ylabel("petal width")