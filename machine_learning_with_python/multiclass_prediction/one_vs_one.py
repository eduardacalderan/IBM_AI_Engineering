import numpy as np
from sklearn import datasets
from sklearn.base import accuracy_score
from sklearn.svm import SVC
import pandas as pd 
from machine_learning_with_python.multiclass_prediction.utils import decision_boundary

pair = [1, 3] # These integers represent the indices of the features you want to select from the dataset.
iris = datasets.load_iris() # This line loads the Iris dataset using the load_iris function from the datasets module.
X = iris.data[:, pair] # This line selects the columns (features) of the Iris dataset specified by the pair list. iris.data is a 2D array where rows represent samples and columns represent features. [:, pair] means "select all rows and only the columns at indices 1 and 3". The result is stored in the variable X.
y = iris.target # This line assigns the target values (class labels) of the Iris dataset to the variable y. iris.target is a 1D array where each element represents the class label of the corresponding sample.

# List  each class
classes_ = set(np.unique(y))

# determine the number of classes 
K = len(classes_)
K*(K-1)/2 #For  𝐾 lasses, we have to train  𝐾(𝐾−1)/2 classifiers. 

# We then train a two-class classifier on each pair of classes. We plot the different training points for each of the two classes.
pairs=[]
left_overs=classes_.copy()
#list used for classifiers 
my_models=[]
#iterate through each class
for class_ in classes_:
  #remove class we have seen before 
  left_overs.remove(class_)
  #the second class in the pair
  for second_class in left_overs:
    pairs.append(str(class_)+' and '+str(second_class))
    print("class {} vs class {} ".format(class_,second_class) )
    temp_y=np.zeros(y.shape)
    #find classes in pair 
    select=np.logical_or(y==class_ , y==second_class)
    #train model 
    model=SVC(kernel='linear', gamma=.5, probability=True)  
    model.fit(X[select,:],y[select])
    my_models.append(model)
    #Plot decision boundary for each pair and corresponding Training samples. 
    decision_boundary (X[select,:],y[select],model,iris,two=True)
    
# plotting the distribution of text length
pairs
majority_vote_array=np.zeros((X.shape[0],3))
majority_vote_dict={}

for j,(model,pair) in enumerate(zip(my_models,pairs)):
  majority_vote_dict[pair]=model.predict(X)
  majority_vote_array[:,j]=model.predict(X)
  
# In the following table, each column is the output of a classifier for each pair of classes and the output is the prediction:
pd.DataFrame(majority_vote_dict).head(10)

# To perform classification on a sample, we perform a majority vote, that is, select the class with the most predictions. We repeat the process for each sample. 
one_vs_one=np.array([np.bincount(sample.astype(int)).argmax() for sample  in majority_vote_array]) 
one_vs_one

# calculate the accuracy:
accuracy_score(y,one_vs_one)