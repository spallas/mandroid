#!/usr/bin/python

import json
import os

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
        data = load_dataset(dataset_path)
        if store:
            with open("drebin_preproc.json", "w") as f:
                json.dump(data, f)

    return preprocess(data)


def preprocess(data):
    return ()


def load_dataset(dataset_path):
    """
    Loads data from Drebin dataset and stores the information of each file in a
    dictionary with the file data + the pair name: file_name

    :param dataset_path: path of the drebin dataset
    :return: list of dictionaries containing files data.
    """
    data = []
    for file in os.listdir(dataset_path):
        # load file data
        data.append(parse_file(dataset_path, file))
    return data


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

    with open(dataset_path+file_name, "r") as f:
        for line in f:
            info = line.strip().split("::")
            if info[0] == "feature":
                if "req_hw" in file_dict:
                    file_dict["req_hw"].append(info[1])
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
                    file_dict["use_urls"].append(info[1])
                else:
                    file_dict["use_urls"] = [info[1]]

    return file_dict
