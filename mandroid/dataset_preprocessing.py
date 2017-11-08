#!/usr/bin/python

import json
import os
import numpy as np
from scipy.sparse import csr_matrix

from sklearn.feature_extraction import DictVectorizer

"""
    This package is responsible of loading the required data and to pre-process
    them before storing them back to JSON format.
    There are two different outputs: one for the classification routines and 
    one for the detection routines. The first one takes into account malicious 
    code only.
"""


def fetch_data(dataset_path, store=False, from_json=False):
    if from_json:
        with open("drebin_preproc.json", "r") as f:
            data = json.load(f)
    else:
        data, hashes = load_dataset(dataset_path)
        if store:
            with open("drebin_preproc.json", "w") as f:
                json.dump(data, f)

    return preprocess(data, hashes)


def preprocess(data, hashes, classify=False):
    """
    Vectorize data and load labels from Drebin dataset .csv file containing
    malware hashes with respective
    :param classify: default = False, if false output data for malware detection
                    otherwise load information about samples classes
    :param data: list of samples as extracted from dataset.
    :param hashes: ashes corresponding to data files.
    :param classify: default False, preprocess for classification, for detection if false.
    :return:
    """
    if classify:
        #TODO
        y = []
    else:
        y = np.zeros(len(data), dtype=np.int8)
        #TODO:

    return vectorize(data), csr_matrix(y)


def load_dataset(dataset_path):
    """
    Loads data from Drebin dataset and stores the information of each file in a
    dictionary with the file data + the pair name: file_name

    :param dataset_path: path of the drebin dataset
    :return: list of dictionaries containing files data.
    """
    data = []
    hash = []
    i=0
    for file in os.listdir(dataset_path):
        if i >= 200:
            break
        # load file data
        data.append(parse_file(dataset_path, file))
        hash.append(file)
        i = i+1
    return data, hash


def parse_file(dataset_path, file_name):
    """
    Build dictionary with 8 features + name:
    0 - apk_hash  : SHA1 of the apk file as name
    1 - req_hw    : requested hardware components
    2 - req_perm  : requested permissions (eg. contacts access)
    3 - app_comp  : components, eg. activities, services
    4 - filt_int  : filtered intents
    5 - rstr_api  : restricted API calls that require a permission
    6 - use_perm  : effectively used permissions
    7 - susp_api  : suspicious API calls
    8 - use_urls  : used network addresses embedded in the code
    The first 4 features were taken from the manifest.xml file while the others
    from the disassembled code.

    :param dataset_path: path of dataset
    :param file_name: name of the file in the format of SHA1 of the apk
    :return: dictionary of features.
    """
    file_dict = {}
    with open(dataset_path + "\\" + file_name, "r") as f:
        for line in f:
            file_dict[line.strip()] = True
            # info = line.strip().split("::")
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
