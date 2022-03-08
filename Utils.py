import numpy as np
import gzip
import math
import time
import pandas as pd
import matplotlib.pyplot as plt
from enum import Enum


class Normalization(Enum):
    STD = 1,
    MIN_MAX = 2,
    NO_NORM = 3


class TrainMethod(Enum):
    BATCH = 1,
    STOCHASTIC = 2,
    OTHER = 3


def generate_permutations_below_target(arr, target, current, temp, i):
    r = np.empty(0)
    if i > 0 and current != target:
        for num in arr:
            if current + num <= target:
                temp_new = np.append(temp, num)
                r = np.append(r, generate_permutations_below_target(arr, target, current + num, temp_new, i - 1))
        return r
    else:
        if i != 0:
            for i in range(i):
                temp = np.append(temp, 0)
        return np.append(r, temp)


def generate_permutations_matrix(max_deg, no_x):
    x = generate_permutations_below_target(np.arange(0, max_deg+1), max_deg, 0, np.empty(0), no_x)
    return x.reshape(-1, no_x)


def generate_polynomial_features(max_deg, x):
    m = generate_permutations_matrix(max_deg, x.shape[1])
    ret = np.ones((x.shape[0], m.shape[0]))
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            po =  np.power(x[:, j].reshape(x.shape[0], 1), m[i, j])
            ret[:, i] = np.multiply(ret[:, i].reshape(ret.shape[0], 1), po).T

    return ret


def load_training_images(file):
    with gzip.open(file, 'r') as f:
        # first 4 bytes is a magic number
        magic_number = int.from_bytes(f.read(4), 'big')
        # second 4 bytes is the number of images
        image_count = int.from_bytes(f.read(4), 'big')
        # third 4 bytes is the row count
        row_count = int.from_bytes(f.read(4), 'big')
        # fourth 4 bytes is the column count
        column_count = int.from_bytes(f.read(4), 'big')
        # rest is the image pixel data, each pixel is stored as an unsigned byte
        # pixel values are 0 to 255
        image_data = f.read()
        images = np.frombuffer(image_data, dtype=np.uint8)\
            .reshape((image_count, row_count, column_count))
        return images


def load_training_labels(file):
    with gzip.open(file, 'r') as f:
        # first 4 bytes is a magic number
        magic_number = int.from_bytes(f.read(4), 'big')
        # second 4 bytes is the number of labels
        label_count = int.from_bytes(f.read(4), 'big')
        # rest is the label data, each label is stored as unsigned byte
        # label values are 0 to 9
        label_data = f.read()
        labels = np.frombuffer(label_data, dtype=np.uint8)
        return labels

