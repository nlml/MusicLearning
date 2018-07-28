# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import h5py
import os
import numpy as np
import errno

from imageFilesTools import getImageData
from config import datasetPath
from config import slicesPath


# Creates name of dataset from parameters
def getDatasetName(nbPerGenre, sliceSize):
    name = "{}".format(nbPerGenre)
    name += "_{}".format(sliceSize)
    return name


# Creates or loads dataset if it exists
# Mode = "train" or "test"
def getDataset(nbPerGenre, genres, sliceSize, validationRatio, testRatio, mode):
    print("[+] Dataset name: {}".format(getDatasetName(nbPerGenre, sliceSize)))
    if not os.path.isfile(datasetPath + "train_X_" + getDatasetName(nbPerGenre, sliceSize) + ".p"):
        print("[+] Creating dataset with {} slices of size {} per genre... ⌛️".format(nbPerGenre, sliceSize))
        createDatasetFromSlices(
            nbPerGenre, genres, sliceSize, validationRatio, testRatio)
    else:
        print("[+] Using existing dataset")

    return loadDataset(nbPerGenre, genres, sliceSize, mode)


# Loads dataset
# Mode = "train" or "test"
def loadDataset(nbPerGenre, genres, sliceSize, mode):
    # Load existing
    datasetName = getDatasetName(nbPerGenre, sliceSize)
    if mode == "train":
        print("[+] Loading training and validation datasets... ")
        train_X = h5fload("{}train_X_{}.p".format(datasetPath, datasetName))
        train_y = h5fload("{}train_y_{}.p".format(datasetPath, datasetName))
        validation_X = h5fload(
            "{}validation_X_{}.p".format(datasetPath, datasetName))
        validation_y = h5fload(
            "{}validation_y_{}.p".format(datasetPath, datasetName))
        print("    Training and validation datasets loaded! ✅")
        return train_X, train_y, validation_X, validation_y

    else:
        print("[+] Loading testing dataset... ")
        test_X = h5fload("{}test_X_{}.p".format(datasetPath, datasetName))
        test_y = h5fload("{}test_y_{}.p".format(datasetPath, datasetName))
        print("    Testing dataset loaded! ✅")
        return test_X, test_y


# Saves dataset
def saveDataset(train_X, train_y, validation_X, validation_y, test_X, test_y, nbPerGenre, genres, sliceSize):
    # Create path for dataset if not existing
    if not os.path.exists(os.path.dirname(datasetPath)):
        try:
            os.makedirs(os.path.dirname(datasetPath))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

    # SaveDataset
    print("[+] Saving dataset... ")
    datasetName = getDatasetName(nbPerGenre, sliceSize)

    h5fdump(train_X, "{}train_X_{}.p".format(datasetPath, datasetName))
    h5fdump(train_y, "{}train_y_{}.p".format(datasetPath, datasetName))
    h5fdump(validation_X, "{}validation_X_{}.p".format(datasetPath, datasetName))
    h5fdump(validation_y, "{}validation_y_{}.p".format(datasetPath, datasetName))
    h5fdump(test_X, "{}test_X_{}.p".format(datasetPath, datasetName))
    h5fdump(test_y, "{}test_y_{}.p".format(datasetPath, datasetName))
    print("    Dataset saved! ✅💾")


def h5fdump(data, path):
    with h5py.File(path, 'w') as hf:
        hf.create_dataset("data", data=data)


def h5fload(path):
    with h5py.File(path, 'r') as hf:
        data = hf['data'][:]
    return data


def get_nb_train_test_val(X, validationRatio, testRatio):
    validationNb = int(len(X) * validationRatio)
    testNb = int(len(X) * testRatio)
    trainNb = len(X) - (validationNb + testNb)
    return trainNb, testNb, validationNb


def isin_idxs(array_to_look_from, array_to_look_in):
    return np.array([array_to_look_in == i for i in array_to_look_from]).sum(0).astype(bool)


def partition_into_train_val_test(rng, filenames, validationRatio, testRatio):
    # Partition such that train, val and test all contain unique songs
    song_idxs = np.array([i.split('_')[1] for i in filenames])
    song_idxs_u = np.array(list(set(song_idxs)))
    nb_valid = max(1, int(len(song_idxs_u) * validationRatio))
    nb_test = max(1, int(len(song_idxs_u) * testRatio))
    nb_train = len(song_idxs_u) - nb_test - nb_valid
    assert nb_train >= 1
    rng.shuffle(song_idxs_u)
    song_idxs_train = song_idxs_u[:nb_train]
    song_idxs_valid = song_idxs_u[nb_train:(nb_train + nb_valid)]
    song_idxs_test = song_idxs_u[(nb_train + nb_valid):]

    train_X = np.array(filenames)[isin_idxs(song_idxs_train, song_idxs)]
    valid_X = np.array(filenames)[isin_idxs(song_idxs_valid, song_idxs)]
    test_X = np.array(filenames)[isin_idxs(song_idxs_test, song_idxs)]
    return train_X, valid_X, test_X


# Creates and save dataset from slices
def createDatasetFromSlices(nbPerGenre, genres, sliceSize, validationRatio, testRatio, seed=1):

    datasetName = getDatasetName(nbPerGenre, sliceSize)

    rng = np.random.RandomState(seed)

    data = {}
    filenames = {}

    # Get all the filenames for every genre and partition into train valid test
    for genre in genres:
        # Get slices in genre subfolder
        filenames[genre] = list(sorted(os.listdir(slicesPath + genre)))
        filenames[genre] = [
            filename for filename in filenames[genre] if filename.endswith('.png')]
        filenames[genre] = filenames[genre][:nbPerGenre]
        # Randomize file selection for this genre
        rng.shuffle(filenames[genre])
        filenames[genre] = dict(zip(['train', 'validation', 'test'],
                                    partition_into_train_val_test(rng, filenames[genre], 0.4, 0.1)))

    # Read images and create / save dataset from train/val/test filename partitions
    for tvt in ['train', 'validation', 'test']:
        print("-> Creating {} set...".format(tvt))
        Xs, ys = [], []
        if tvt not in data:
            data[tvt] = []
        for genre in genres:
            print("->-> Adding {}...".format(genre))
            # Add data (X,y)
            X = np.array([getImageData(os.path.join(slicesPath + genre, filename), sliceSize)
                          for filename in filenames[genre][tvt]]).astype(np.float32)
            single_y = np.array(
                [1. if genre == g else 0. for g in genres]).reshape(1, -1)
            y = np.tile(single_y.astype(np.float32), [
                        len(filenames[genre][tvt]), 1])
            Xs.append(X)
            ys.append(y)

        # Concatenate over genres
        X, y = (np.concatenate(Xs, 0).astype(np.float32),
                np.concatenate(ys, 0).astype(np.float32))

        # Random shuffle X and y
        perm = rng.permutation(len(X))
        X = X[perm]
        y = y[perm]

        if not os.path.exists(datasetPath):
            os.makedirs(datasetPath)

        # Save dataset to disk
        h5fdump(X, "{}{}_X_{}.p".format(datasetPath, tvt, datasetName))
        h5fdump(y, "{}{}_y_{}.p".format(datasetPath, tvt, datasetName))
