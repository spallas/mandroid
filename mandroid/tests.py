#!/usr/bin/python
import time
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score

from mandroid.dataset_preprocessing import fetch_data
from mandroid.mdetect import get_clf

"""
    File with tests for correctness and performance evaluations
"""


def main():

    profiling = True
    k = 10
    print("Fetching data from... ")
    begin = time.time()

    X, y = fetch_data("/Users/davidespallaccini/developer/learning/datasets/drebin/feature_vectors",
                      max_samples=10000)
    if profiling:
        print("Time for fetching data:\t" + str(time.time() - begin))
        begin = time.time()

    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.3, random_state=1)

    model = get_clf("SVM").fit(X_train, Y_train)
    if profiling:
        print("Time for training network:\t" + str(time.time() - begin))

    Y_predicted = model.predict(X_test)

    # Confusion matrix results
    conf_mat = confusion_matrix(Y_predicted, Y_test, [0, 1])
    print("true negative\tfalse negative")
    print("false positive\ttrue positive")
    print(conf_mat)

    begin = time.time()
    # K-fold cross validation results
    validation_results = cross_val_score(get_clf("SVM"), X, y, cv=k)
    print(validation_results)
    print("Min/avg/max from K-fold:\t"
          + str(min(validation_results)) + ", "
          + str(sum(validation_results)/float(len(validation_results))) + ", "
          + str(max(validation_results)))

    if profiling:
        print("Time for " + str(k) + "-fold cross validation:\t" + str(time.time() - begin))


if __name__ == '__main__':
    main()
