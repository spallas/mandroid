#!/usr/bin/python
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC

"""
    This module is able to check is a given application is an Android malware.
"""

svm_clf = LinearSVC(random_state=0)
nb_clf = GaussianNB()


def get_clf(name):
    if name == "SVM":
        return svm_clf
    elif name == "Bayes":
        return nb_clf

