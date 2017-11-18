#!/usr/bin/python


import csv
import os
import pickle

import numpy as np
from sklearn.feature_extraction import DictVectorizer

"""
    This package is responsible of loading the required data and to pre-process
    them before storing them back to JSON format.
    There are two different outputs: one for the classification routines and 
    one for the detection routines. The first one takes into account malicious 
    code only.
"""


def fetch_data(dataset_path, max_samples=1000, malware_percentage=35, store=False, from_store=False):
    """
    Load data from dataset or from saved result stored as JSON file. If store id True
    the loaded dataset will be stored in a JSON file.
    :param malware_percentage: Number of samples loaded for training
    :param max_samples:
    :param dataset_path: Position of the dataset folder
    :param store: set it to True if you want to save the file parsing to JSON file
    :param from_store: load data from JSON file instead of from original dataset files
    :return:
    """
    if from_store:
        with open("drebin_preproc.pkl", "rb") as f:
            data = pickle.load(f)
            classes = pickle.load(f)
    else:
        malware_path = dataset_path[:dataset_path.rfind('/')+1] + "sha256_family.csv"
        data, classes = load_dataset(dataset_path, malware_path, max_samples, malware_percentage)
        data, classes = preprocess(data, classes)
        if store:
            with open("drebin_preproc.pkl", "wb") as f:
                pickle.dump(data, f)
                pickle.dump(classes, f)

    return data, classes


def preprocess(data, y):
    """
    Vectorize data and shuffle rows with corresponding classes.
    Converts data collected from dataset to a sparse vector
    suitable for the SVM primitives.

    :param data: array of dictionaries corresponding to dataset files with values
                of the type -> "feature=feature_value": True/False
    :param y: a numpy array containing samples classes
    :return: a sparse vector composed of a 1 corresponding to present features
            0 otherwise
    """

    vectorizer = DictVectorizer()
    return vectorizer.fit_transform(data), y  # csr_matrix(y)


def load_dataset(dataset_path, malware_file_path, max_samples, percentage_malware):
    """
    Loads data from Drebin dataset and stores the information of each file in a
    dictionary with the file data + the pair name: file_name

    :param dataset_path: path of the Drebin dataset
    :return: list of dictionaries containing files data.
    """
    n_malware = int(max_samples*(percentage_malware/100))
    if n_malware > 5560:
        # there are only 5560 malware files in the Drebin dataset
        n_malware = 5560
    n_goodware = int(max_samples - n_malware)
    data = []
    Y = np.zeros(n_malware + n_goodware)
    i = 0
    malware_hash = load_malware(malware_file_path)

    print("n_malware " + str(n_malware) + ", n_goodware " + str(n_goodware))

    for file in os.listdir(dataset_path):
        # load file data
        if n_malware + n_goodware > 0:
            if file in malware_hash:
                if n_malware > 0:
                    data.append(parse_file(dataset_path, file))
                    Y[i] = 1
                    n_malware -= 1
                    i += 1
            else:
                if n_goodware > 0:
                    data.append(parse_file(dataset_path, file))
                    Y[i] = 0
                    n_goodware -= 1
                    i += 1
        else:
            break
    return data, Y


def parse_file(dataset_path, file_name):
    """
    Create a dictionary of the form {}
    :param dataset_path: path of dataset
    :param file_name: name of the file in the format of SHA1 of the apk
    :return: dictionary of features.
    """
    file_dict = {}
    with open(os.path.join(dataset_path, file_name)) as f:
        for line in f:
            file_dict[line.strip()] = True
    return file_dict


def load_malware(malware_file_path):
    """
    Load the .csv containing:
        sha1 of the apk file | malware family
    :param malware_file_path: path of the malware file
    :return: dictionary with sha1 of apk as keys and True as values
            (each sample in the file is a malware)
    """
    malware_hash = {}
    with open(malware_file_path, newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            malware_hash[row[0]] = True

    return malware_hash
