import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pandas as pd

plot_colors = "ryb"
plot_step = 0.02

# This function plot a different decicion boundary
def decision_boundary(X, y, model, iris, two=None):
  x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1 # extracting the minimum and maximum values from the first column of a NumPy array X, adjusting these values to create a margin, and then storing the results in variables. X[:, 0]: X is assumed to be a NumPy array, and X[:, 0] selects all the rows in the first column of X. The colon (:) indicates that we are selecting all rows, and 0 specifies the first column.
  y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
  xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step), np.arange(y_min, y_max, plot_step)) # np.meshgrid: Create a rectangular grid out of an array of x values and an array of y values. np.arange: Return evenly spaced values within a given interval.

  plt.tight_layout(h_pad=0.5, w_pad=0.5, pad=2.5) # plt.tight_layout: Automatically adjust subplot parameters to give specified padding.
  
  Z = model.predict(np.c_[xx.ravel(), yy.ravel()]) # stores the predictions made by the model. The predictions could be class labels, probabilities, or any other output depending on the type of model used.
  Z = Z.reshape(xx.shape) # reshapes the predictions to match the shape of the grid created by np.meshgrid. This is necessary because the predictions are made on a grid, and we need to reshape them to plot the decision boundary.
  cs = plt.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu) # plt.contourf: Create a filled contour plot. xx, yy: The x and y coordinates of the grid. Z: The values of the grid. cmap: The color map to use for the contour plot.
  
  # This code creates different types of plots based on the value of two. If two is True, it creates a filled contour plot and scatter plots for unique values in y. If two is False, it creates scatter plots for values in y and iris.target with different markers and colors.
  if two:
    cs = plt.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu) # Creates a filled contour plot using the xx, yy, and Z arrays with the colormap RdYlBu
    for i, color in zip(np.unique(y), plot_colors): # Iterates over unique values in y and corresponding colors in plot_colors.
      idx = np.where(y == i)
      plt.scatter(X[idx, 0], X[idx, 1], label=y, cmap=plt.cm.RdYlBu, S=15) # Plots a scatter plot for the points in X where y equals i. The color of the points is determined by the colormap RdYlBu.
    plt.show()
  else:
    set_={0,1,2}
    print(set_)
    for i, color in zip(range(3), plot_colors): #  Iterates over the range 0 to 2 and corresponding colors in plot_colors.
      idx = np.where(y == i)
      if np.any(idx):
        set_.remove(i)
        plt.scatter(X[idx, 0], X[idx, 1], label=y, cmap=plt.cm.RdYlBu, edgecolor='black', s=15)
        
    for i in set_:
      idx = np.where(iris.target == i)
      plt.scatter(X[idx, 0], X[idx, 1], marker='x', color='black')
    plt.show()


# This function will plot the probability of belonging to each class; Each column is the probability of belonging to na class and the row number is the sample number
def plot_probability_array(X, probability_array):
  # Initialize a zero array for plotting. The array has the same number of rows as X and 30 columns.
  plot_array = np.zeros((X.shape[0], 30))
  col_start = 0
  # Iterate over the classes and their corresponding column end indices
  for class_, col_end in enumerate([10, 20, 30]):
      # Fill the plot_array with repeated probabilities for the current class
      plot_array[:, col_start:col_end] = np.repeat(probability_array[:, class_].reshape(-1, 1), 10, axis=1)
      # Update the starting column index for the next class
      col_start = col_end
  # Display the plot array as an image
  plt.imshow(plot_array)
  # Remove x-axis ticks
  plt.xticks([])
  # Set y-axis label
  plt.ylabel(["samples"])
  # Set x-axis label
  plt.xlabel(["probability of 3 classes"])
  # Add a colorbar to the plot
  plt.colorbar()
  # Show the plot
  plt.show()