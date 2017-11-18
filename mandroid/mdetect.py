#!/usr/bin/python
import pickle

import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import learning_curve

"""
    This module is able to check is a given application is an Android malware.
"""

svm_clf = LinearSVC(tol=1e-4, max_iter=1000, random_state=1)
sgd_clf = SGDClassifier(tol=1e-4, max_iter=1000, alpha=0.0001, loss='modified_huber')
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


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and training learning curve.
    :param train_sizes: default
    :param estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.
    :param title : string Title for the chart.
    :param X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.
    :param y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.
    :param ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.
    :param cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - An object to be used as a cross-validation generator.
          - An iterable yielding train/test splits.
        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.
        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.
    :param n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt
