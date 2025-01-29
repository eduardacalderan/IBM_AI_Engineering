# For one-vs-all classification, if we have K classes, we use K two-class classifier models. The number of class labels present in the dataset is equal to the number of generated classifiers. First, we create an artificial class we will call this "dummy" class. For each classifier, we split the data into two classes. We take the class samples we would like to classify, the rest of the samples will be labelled as a dummy class. We repeat the process for each class. To make a classification, we use the classifier with the highest probability, disregarding the dummy class.

# Here, we train three classifiers and place them in the list my_models. For each class we take the class samples we would like to classify, and the rest will be labelled as a dummy class. We repeat the process for each class. For each classifier, we plot the decision regions. The class we are interested in is in red, and the dummy class is in blue. Similarly, the class samples are marked in blue, and the dummy samples are marked with a black x.
from sklearn import datasets
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

from utils import decision_boundary, plot_probability_array

pair = [1, 3] # These integers represent the indices of the features you want to select from the dataset.
iris = datasets.load_iris() # This line loads the Iris dataset using the load_iris function from the datasets module.
X = iris.data[:, pair] # This line selects the columns (features) of the Iris dataset specified by the pair list. iris.data is a 2D array where rows represent samples and columns represent features. [:, pair] means "select all rows and only the columns at indices 1 and 3". The result is stored in the variable X.
y = iris.target 


# dummy class
dummy_class = y.max() + 1

# list used for classifiers 
my_models = []

# iterate through each class
for class_ in np.unique(y):
  #s select the index of our class
  select=(y==class_)
  temp_y=np.zeros(y.shape)
  
  # class, we are trying to classify
  temp_y[y==class_]=class_
  
  # set other samples to a dummy class
  temp_y[y!=class_]=dummy_class
  
  # train model and add to list 
  model = SVC(kernel='linear', gamma=0.5, probability=True)
  my_models.append(model.fit(X, temp_y))
  
  # plot decision boundary
  decision_boundary(X,temp_y,model,iris)

#  For each sample we calculate the  probability of belonging to each class, not including the dummy class.
probability_array = np.zeros((X.shape[0], 3))
for j, model in enumerate(my_models):
  real_class = np.where(np.array(model.classes_) !=3 )[0]
  probability_array[:, j] = model.predict_proba(X)[:, real_class][:, 0]
  
probability_array[0, :]
print(probability_array[0, :])

# We can plot the probability of belonging to the class. The row number is the sample number.
plot_probability_array(X,probability_array)

# We can apply the argmax function to each sample to find the class.
one_vs_all = np.argmax(probability_array, axis=1)
one_vs_all
print(one_vs_all)

# We can calculate the accuracy. 
accuracy_score(y, one_vs_all)
