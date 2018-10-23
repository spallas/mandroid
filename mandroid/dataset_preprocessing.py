#!/usr/bin/python


import csv
import os
import pickle

import numpy as np
from sklearn.feature_extraction import DictVectorizer

"""
    This package is responsible of loading the required data and to pre-process
    them before storing them back to Pickle binary format.
    There are two different outputs: one for the classification routines and 
    one for the detection routines. The first one takes into account malicious 
    code only.
"""


def fetch_data(dataset_path, max_samples=1000, malware_percentage=35, 
               store=False, from_store=False, hashes_path="sha256_family.csv",
               features_path="feature_vectors"):
    """
    Load data from dataset or from saved result stored as Pickle file. If store id True
    the loaded dataset will be stored in a Pickle file.
    :param malware_percentage: Number of samples loaded for training. Ignored if from_store=True
    :param max_samples: Total number of samples. Ignored if from_store=True
    :param dataset_path: Position of the dataset folder
    :param store: set it to True if you want to save the file parsing to JSON file
    :param from_store: load data from JSON file instead of from original dataset files
    :return:
    """
    if from_store:
        with open("drebin_preproc.pkl", "rb") as f:
            X = pickle.load(f)
            Y = pickle.load(f)
    else:
        malware_path = os.path.join(dataset_path, hashes_path)
        feature_vectors_path = os.path.join(dataset_path, features_path)
        X, Y = load_dataset(feature_vectors_path, malware_path, max_samples, malware_percentage)
        if store:
            with open("drebin_preproc.pkl", "wb") as f:
                pickle.dump(X, f)
                pickle.dump(Y, f)

    return X, Y


def load_dataset(dataset_path, malware_file_path, max_samples, percentage_malware):
    """
    Loads data from Drebin dataset and stores the information of each file in a
    dictionary with the file data + the pair name: file_name

    :param dataset_path: path of the Drebin dataset
    :return: list of dictionaries containing files data.
    """
    vectorizer = DictVectorizer()
    n_malware = int(max_samples*(percentage_malware/100))
    if n_malware > 5560:  # there are only 5560 malware files in the Drebin dataset
        n_malware = 5560
    n_goodware = int(max_samples - n_malware)
    data = []
    malware_hashes = set()
    with open(malware_file_path, newline='') as f:
        for row in csv.reader(f):
            malware_hashes.add(row[0])
    print("n_malware " + str(n_malware) + ", n_goodware " + str(n_goodware))
    Y = np.zeros(n_malware + n_goodware)
    i = 0
    for hash_name in os.listdir(dataset_path):
        if n_malware + n_goodware > 0:
            if hash_name in malware_hashes:
                if n_malware > 0:
                    data.append(parse_file(dataset_path, hash_name))
                    Y[i] = 1
                    n_malware -= 1
                    i += 1
            else:
                if n_goodware > 0:
                    data.append(parse_file(dataset_path, hash_name))
                    Y[i] = 0
                    n_goodware -= 1
                    i += 1
        else:
            break
    return vectorizer.fit_transform(data), Y


def parse_file(dataset_path, file_name, separator="::"):
    """
    Create a dictionary of the form \{ feature_name: string \}
    :param dataset_path: path of dataset
    :param file_name: name of the file in the format of SHA1 of the apk
    :return: dictionary of features.
    """
    features = {}
    with open(os.path.join(dataset_path, file_name)) as f:
        for line in f:
            if separator in line:
                feature_name, string = line.strip().split(separator, maxsplit=1)
                features[feature_name] = string
    return features
