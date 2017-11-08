#!/usr/bin/python

import json
import os
import numpy as np
from scipy.sparse import csr_matrix
import csv

from sklearn.feature_extraction import DictVectorizer

"""
    This package is responsible of loading the required data and to pre-process
    them before storing them back to JSON format.
    There are two different outputs: one for the classification routines and 
    one for the detection routines. The first one takes into account malicious 
    code only.
"""


def fetch_data(dataset_path, store=False, from_json=False):
    """
    Load data from dataset or from saved result stored as JSON file. If store id True
    the loaded dataset will be stored in a JSON file.
    :param dataset_path: Position of the dataset folder
    :param store: set it to True if you want to save the file parsing to JSON file
    :param from_json: load data from JSON file instead of from original dataset files
    :return:
    """
    if from_json:
        with open("drebin_preproc.json", "r") as f:
            data = json.load(f)
    else:
        data, hashes = load_dataset(dataset_path)
        if store:
            with open("drebin_preproc.json", "w") as f:
                json.dump(data, f)

    return preprocess(data, hashes)


def preprocess(data, y, classify=False):
    """
    Vectorize data and shuffle rows with corresponding classes.
    :param data: list of samples as extracted from dataset.
    :param y: a numpy array containing samples classes
    :return:
    """
    toshuffle = np.column_stack((data, y))
    np.random.shuffle(toshuffle)
    result = np.hsplit(toshuffle, np.array([1]))

    data, y = result[0].transpose()[0], result[1].transpose()[0]
    return vectorize(data), csr_matrix(y)


def load_dataset(dataset_path, malware_file_path, max_samples, percentage_malware=35):
    """
    Loads data from Drebin dataset and stores the information of each file in a
    dictionary with the file data + the pair name: file_name

    :param dataset_path: path of the drebin dataset
    :return: list of dictionaries containing files data.
    """
    n_malware = int(max_samples*(percentage_malware/100))
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
    with open(dataset_path + "\\" + file_name, "r") as f:
        for line in f:
            file_dict[line.strip()] = True
    return file_dict


def vectorize(data):
    """
    Converts data collected from dataset to a sparse vector
    suitable for the SVM primitives.

    :param data: array of dictionaries corresponding to dataset files with values
                of the type -> "feature=feature_value": True/False
    :return: a sparse vector composed of a 1 corresponding to present features
            0 otherwise
    """
    vectorizer = DictVectorizer()
    return vectorizer.fit_transform(data)


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


#X, Y = load_dataset("C:\\Users\\Valerio\\Downloads\\Machine Learning\\HW\\drebin\\feature_vectors", "C:\\Users\\Valerio\\Downloads\\Machine Learning\\HW\\drebin\\sha256_family.csv", 30)
