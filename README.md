# Mandroid
Automatic Android malware detection and classification using various machine learning techniques.

```
usage: tests.py [-h] [-t CLASSIFIER] [-p] [-plt] [-f]
                [-k 3-20]
                dataset_path

positional arguments:
  dataset_path          Path of the folder containing the Drebin files

optional arguments:
  -h, --help            show this help message and exit
  -t CLASSIFIER,
  --train CLASSIFIER    Available classifiers: SVM, SGD, (NBayes, NeuralNet deprecated)
  -p, --profile         Print timing info of program steps
  -plt, --plot          Plot learning curve as evaluation technique
  -f, --fast_load       Load data from saved .pkl file, fails if file not
                        existing
  -k 3-20               Use this K for K-fold cross validation

Process finished with exit code 0

```
