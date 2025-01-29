from sklearn import datasets
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

from utils import decision_boundary


# Sklean performs Multi-class Classification automatically, we can apply the method and calculate the accuracy. Train a SVM classifier with the `kernel` set to `linear`, `gamma` set to `0.5`, and the `probability` paramter set to `True`, then train the model using the `X` and `y` data.

pair = [1, 3] # These integers represent the indices of the features you want to select from the dataset.
iris = datasets.load_iris() # This line loads the Iris dataset using the load_iris function from the datasets module.
X = iris.data[:, pair] # This line selects the columns (features) of the Iris dataset specified by the pair list. iris.data is a 2D array where rows represent samples and columns represent features. [:, pair] means "select all rows and only the columns at indices 1 and 3". The result is stored in the variable X.
y = iris.target 

model = SVC(kernel='linear', gamma=0.5, probability=True)
model.fit(X, y)

yhat = model.predict(X)
accuracy_score(y, yhat) 

decision_boundary (X,y,model,iris)