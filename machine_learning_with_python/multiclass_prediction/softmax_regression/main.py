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

plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu) #  creates a scatter plot using the first two features of the dataset X. The points are colored based on their class labels y, and the colormap RdYlBu is used to differentiate the classes.
plt.xlabel("sepal width (cm)") # sets the label for the x-axis of the plot to "sepal width (cm)".
plt.ylabel("petal width") # sets the label for the y-axis of the plot to "petal width".

# Softmax Regression
lr = LogisticRegression(random_state=0).fit(X, y) # Initializes a LogisticRegression model with a fixed random state for reproducibility and fits it to the dataset X and labels y.

probability = lr.predict_proba(X) # uses the fitted logistic regression model to predict the class probabilities for each sample in X.

plot_probability_array(X,probability) # calls the plot_probability_array function to visualize the predicted probabilities for each class.

probability[0, :]

probability[0, :].sum()

# applying the argmax function
np.argmax(probability[0, :]) # finds the index of the class with the highest predicted probability for the first sample.

# applying the argmax function to each sample
softmax_prediction = np.argmax(probability, axis=1) # applies the argmax function to each sample in the dataset to get the predicted class labels based on the highest probability.
softmax_prediction

# We can verify that sklearn does this under the hood by comparing it to the output of the method predict
yhat = lr.predict(X) # applies the argmax function to each sample in the dataset to get the predicted class labels based on the highest probability.
accuracy_score(yhat, softmax_prediction) # calculates the accuracy score by comparing the predicted class labels from the predict method (yhat) with those obtained from the argmax function (softmax_prediction). This verifies that both methods produce the same predictions.

# We can't use Softmax regression for SVMs, Let's explore two methods of Multi-class Classification that we can apply to SVM.