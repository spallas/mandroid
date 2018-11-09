#!/usr/bin/python
import pickle
import argparse
import time

import numpy as np
import matplotlib.pyplot as plt

from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import learning_curve
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score, ShuffleSplit

from dataset_preprocessing import fetch_data

"""
    This module trains an Android malware binary classifier.
"""


def get_clf(name):
    if name == "SVM":
        return LinearSVC(tol=1e-4, max_iter=1000, random_state=1)
    elif name == "NBayes":
        return GaussianNB()
    elif name == "NeuralNet":
        return MLPClassifier()
    else:  # SGD is default
        return SGDClassifier(tol=1e-4, max_iter=1000, alpha=0.0001, loss='modified_huber')


def train(clf_name, X, y):
    if clf_name == "NBayes":
        X = X.toarray()
    return get_clf(clf_name).fit(X, y)


def store_trained_clf(clf, clf_name, clf_file_name="classifier.pkl"):
    with open(clf_name + "-" + clf_file_name, "wb") as f:
        pickle.dump(clf, f)


def load_trained_clf(clf, clf_name, clf_file_name="classifier.pkl"):
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


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--train", metavar="CLASSIFIER", type=str,
                        help="Available classifiers: SVM, SGD (default), NBayes, NeuralNet")
    parser.add_argument("dataset_path", type=str,
                        help="Path of the folder containing the Drebin files (w/o  '/')")
    parser.add_argument("-p", "--measure_time", action="store_true",
                        help="Print timing info of program steps")
    parser.add_argument("-plt", "--plot", action="store_true",
                        help="Plot learning curve as evaluation technique")
    parser.add_argument("-f", "--fast_load", action="store_true",
                        help="Load data from saved .pkl file, fails if file not existing")
    parser.add_argument("-k", type=int, default=10, choices=range(4, 20),
                        help="Use this K for K-fold cross validation")
    args = parser.parse_args()

    measure_time = args.measure_time
    k = args.k
    dataset_path = args.dataset_path
    clf_name = args.train

    if args.fast_load:
        print("Fetching data from pickle serialization...")
    else:
        print("Fetching data from " + dataset_path)
    begin = time.time()

    X, y = fetch_data(dataset_path, max_samples=123000, malware_percentage=5, 
                      from_store=args.fast_load, store=True)
    if measure_time:
        print("Time for fetching data:\t" + str(time.time() - begin))
        begin = time.time()
    if args.plot:
        cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
        plot_learning_curve(get_clf(clf_name), "Learning curve", X, y, cv=cv)
        
        if measure_time:
            print("Time for building learning curve:\t" + str(time.time() - begin))
        plt.show()

    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    
    model = train(clf_name, X_train, Y_train)
    
    if measure_time:
        print("Time for training network:\t" + str(time.time() - begin))

    if clf_name == "NBayes":
        X_test = X_test.toarray()

    Y_predicted = model.predict(X_test)

    # Confusion matrix results
    print("Confusion matrix results on test split consisting of 20% of total data:")
    conf_mat = confusion_matrix(Y_predicted, Y_test, [0, 1])
    print("true negative\tfalse negative")
    print("false positive\ttrue positive")
    print(conf_mat)

    begin = time.time()
    # K-fold (stratified) cross validation results
    if clf_name == "NBayes":
        X = X.toarray()
    print("K-fold cross validation results:")
    validation_results = cross_val_score(get_clf(clf_name), X, y, cv=k)
    print(validation_results)
    print("Min/avg/max from K-fold:\t"
          + str(min(validation_results)) + ", "
          + str(sum(validation_results)/float(len(validation_results))) + ", "
          + str(max(validation_results)))

    if measure_time:
        print("Time for " + str(k) + "-fold cross validation:\t" + str(time.time() - begin))


if __name__ == '__main__':
    main()
