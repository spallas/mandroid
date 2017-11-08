#!/usr/bin/python

from sklearn.svm import LinearSVC

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
    clf.fit(X, y)
    return
