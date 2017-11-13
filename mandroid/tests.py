#!/usr/bin/python
import argparse
import time
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score

from mandroid.dataset_preprocessing import fetch_data
from mandroid.mdetect import train, get_clf

"""
    File with tests for correctness and performance evaluations
"""


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--train", metavar="CLASSIFIER",
                        type=str, help="Available classifiers: SVM, SGD, NBayes, NeuralNet")
    parser.add_argument("dataset_path", type=str, help="Path of the folder containing the Drebin files (w/o  '/')")
    parser.add_argument("-p", "--profile", action="store_true", help="Print timing info of program steps")
    parser.add_argument("-k", type=int, default=10, choices=range(4, 20), help="Use this K for K-fold cross validation")
    args = parser.parse_args()

    profiling = args.profile
    k = args.k
    dataset_path = args.dataset_path
    clf_name = args.train

    print("Fetching data from... ")
    begin = time.time()

    X, y = fetch_data(dataset_path, max_samples=50000, malware_percentage=10)
    if profiling:
        print("Time for fetching data:\t" + str(time.time() - begin))
        begin = time.time()

    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    model = train(clf_name, X_train, Y_train)
    if profiling:
        print("Time for training network:\t" + str(time.time() - begin))

    if clf_name == "NBayes":
        X_test = X_test.toarray()

    Y_predicted = model.predict(X_test)

    # Confusion matrix results
    conf_mat = confusion_matrix(Y_predicted, Y_test, [0, 1])
    print("true negative\tfalse negative")
    print("false positive\ttrue positive")
    print(conf_mat)

    begin = time.time()
    # K-fold (stratified) cross validation results
    if clf_name == "NBayes":
        X = X.toarray()
    validation_results = cross_val_score(get_clf(clf_name), X, y, cv=k)
    print(validation_results)
    print("Min/avg/max from K-fold:\t"
          + str(min(validation_results)) + ", "
          + str(sum(validation_results)/float(len(validation_results))) + ", "
          + str(max(validation_results)))

    if profiling:
        print("Time for " + str(k) + "-fold cross validation:\t" + str(time.time() - begin))


if __name__ == '__main__':
    main()
