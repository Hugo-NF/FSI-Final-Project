import numpy as np
from glob import glob
import sklearn.datasets as reader

from sklearn import model_selection
from sklearn.neighbors import KNeighborsClassifier
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# Read .svm files
filenames = glob("../data/features_split/*.svm")
data = reader.load_svmlight_files(filenames)

# Features, Labels
X, Y = data[0], data[1]
seed = 7
scoring = 'accuracy'

# # Divide between train and test
# X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, y, test_size=0.2, random_state=seed)

# Algoritms and classifiers array
models = [('KNN', KNeighborsClassifier()),
          ('SVM', SVC(gamma='auto'))]

# Run
results = []
names = []
for name, model in models:
    kfold = model_selection.KFold(n_splits=10, random_state=seed)
    cv_results = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)
