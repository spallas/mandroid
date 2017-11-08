#!/usr/bin/python
from functools import reduce

from sklearn import svm
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.svm import LinearSVC
from mandroid.dataset_preprocessing import load_dataset
from mandroid.dataset_preprocessing import vectorize

"""
    This module is able to check is a given application is an Android malware.
"""


def train_and_validate(X, y):
    """
    Train the model with X, y from dataset and evaluate performance with
    10-fold cross validation. Print vali
    :param X:
    :param y:
    :return:
    """
    clf = LinearSVC(random_state=0)
    X = vectorize(X)

    train_test_SVM(X, Y, clf)
    #scores = cross_val_score(clf, X, Y, cv=10)
    #print(scores)
    #clf.fit(X, y)
    return

def train_test_SVM(X, Y, clf):

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=1)
    model = clf.fit(X_train, Y_train)
    result = model.predict(X_test)

    merged_list=list(zip(result,Y_test))
    #f = lambda x, y: if x(0)!=x(1)
    def my_f(x):
        if x[0]!=x[1] : return 1
        return 0
    f=my_f
    errors=reduce(lambda x,y:x+y, map(f,merged_list))
    print(errors)
    print(X.shape)
    #copyright anjel§





X, Y = load_dataset("C:\\Users\\Valerio\\Downloads\\Machine Learning\\HW\\drebin\\feature_vectors",
                        "C:\\Users\\Valerio\\Downloads\\Machine Learning\\HW\\drebin\\sha256_family.csv", 1000, 1)

train_and_validate(X,Y)