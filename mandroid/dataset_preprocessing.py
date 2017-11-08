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
    i=0
    malware_hash = load_malwares(malware_file_path)

    print("n_mwalware " + str(n_malware) + ", n_goodware " + str(n_goodware))

    for file in os.listdir(dataset_path):
        # load file data

        if n_malware + n_goodware > 0:
            if file in malware_hash:
                if n_malware > 0:
                    data.append(parse_file(dataset_path, file))
                    Y[i] = 1
                    n_malware -=1
                    i+=1

            else:
                if n_goodware > 0:
                    data.append(parse_file(dataset_path, file))
                    Y[i] = 0
                    n_goodware -=1
                    i+=1

        else:

            break
    return data, Y


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
            '''if info[0] == "feature":
                            if "req_hw" in file_dict:
                                file_dict[info[0]+"req_hw"+info[1]] = True;
                            else:
                                file_dict["req_hw"] = [info[1]]
                        elif info[0] == "permission":
                            if "req_perm" in file_dict:
                                file_dict["req_perm"].append(info[1])
                            else:
                                file_dict["req_perm"] = [info[1]]
                        elif (info[0] == "activity"
                              or info[0] == "service_receiver"
                              or info[0] == "provider"
                              or info[0] == "service"):
                            if "app_comp" in file_dict:
                                file_dict["app_comp"].append(info[1])
                            else:
                                file_dict["app_comp"] = [info[1]]
                        elif info[0] == "intent":
                            if "filt_int" in file_dict:
                                file_dict["filt_int"].append(info[1])
                            else:
                                file_dict["filt_int"] = [info[1]]
                        elif info[0] == "api_call":
                            if "rstr_api" in file_dict:
                                file_dict["rstr_api"].append(info[1])
                            else:
                                file_dict["rstr_api"] = [info[1]]
                        elif info[0] == "real_permission":
                            if "use_perm" in file_dict:
                                file_dict["use_perm"].append(info[1])
                            else:
                                file_dict["use_perm"] = [info[1]]
                        elif info[0] == "call":
                            if "susp_api" in file_dict:
                                file_dict["susp_api"].append(info[1])
                            else:
                                file_dict["susp_api"] = [info[1]]
                        elif info[0] == "url":
                            if "use_urls" in file_dict:
                                file_dict["use_urls"].append
                                '''
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


def load_malwares(malware_file_path):

    malware_hash = {}
    with open(malware_file_path, newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            malware_hash[row[0]] = True

    return malware_hash


#X, Y = load_dataset("C:\\Users\\Valerio\\Downloads\\Machine Learning\\HW\\drebin\\feature_vectors", "C:\\Users\\Valerio\\Downloads\\Machine Learning\\HW\\drebin\\sha256_family.csv", 30)
