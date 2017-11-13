#!/usr/bin/python
import pickle

from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier

"""
    This module is able to check is a given application is an Android malware.
"""

svm_clf = LinearSVC(random_state=0)
sgd_clf = SGDClassifier(tol=1e-3, max_iter=1000, alpha=0.00007)
nb_clf = GaussianNB()
neural_net_clf = MLPClassifier()

clf_file_name = "classifier.pkl"


def get_clf(name):
    if name == "SVM":
        return svm_clf
    elif name == "NBayes":
        return nb_clf
    elif name == "NeuralNet":
        return neural_net_clf
    elif name == "SGD":
        return sgd_clf


def train(clf_name, X, y):
    if clf_name == "NBayes":
        X = X.toarray()

    return get_clf(clf_name).fit(X, y)


def store_trained_clf(clf, clf_name):
    with open(clf_name + "-" + clf_file_name, "wb") as f:
        pickle.dump(clf, f)


def load_trained_clf(clf, clf_name):
    with open(clf_name + "-" + clf_file_name) as f:
        return pickle.load(clf, f)

