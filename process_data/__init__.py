"""
takes the .mat file, and Images directory, and creates:
    A numpy array of the images as 64x64 greyscale crops.
    A python dictionary for the labels, with the folowing:
        - digit labels
        - filename of the original file.
        - number of digits.
"""
from __future__ import print_function, absolute_import
import os
import numpy as np
import time
import pandas as pd
from support import digit_array_to_int, merge_arrays
from .helpers import mat_data_to_dict_of_arrays, \
    get_full_bounding_boxes, create_images_array, add_digit_bounding_boxes,\
    create_single_bboxes_array, fix_null_digit_bboxes
from file_support import obj2pickle, pickle2obj


# ==============================================================================
#                                                                  DATA_CLEANING
# ==============================================================================
def data_cleaning(matfile, images_dir, dataset, limit=None):
    """
    Args:
        matfile:
        images_dir:
        dataset:        (str) Which one of the datasets is used?
                        value must be one of: "train", "extra" or "test"
        limit: restrict dataset to only process this number of items

    Returns:

    """
    assert dataset.lower() in ["train", "extra",
                               "test"], "Incorrect dataset val"

    # EXTRACT LABEL DATA FROM MAT FILE
    t0 = time.time()
    print("Converting Mat Data to Dict of Arrays")
    print("  ", matfile)
    labels_dict = mat_data_to_dict_of_arrays(matfile, limit=limit)
    print("-- [DONE] ({} seconds)".format(time.time() - t0))

    # ADDING DATASET LABEL
    labels_dict["dataset"] = [dataset] * labels_dict["N"].shape[0]
    labels_dict["dataset"] = np.array(labels_dict["dataset"])

    # BOUNDING BOXES
    print("Processing bounding boxes data", end="")
    labels_dict["bbox"] = get_full_bounding_boxes(labels=labels_dict)

    # ADD DIGIT BOUNDING BOXES
    add_digit_bounding_boxes(labels=labels_dict)

    # MERGE ALL BOUNDING BOXES - into a single bboxes array
    bboxes = ["bbox", "bbox1", "bbox2", "bbox3", "bbox4", "bbox5"]
    labels_dict["bboxes"] = create_single_bboxes_array(labels=labels_dict,
                                                       bboxes=bboxes)
    print("-- [DONE] ({} seconds)".format(time.time() - t0))

    # CREATE IMAGES ARRAY
    array_image_dim = 64  # Square shape of the array images
    print("Creating Loosely cropped images array", end="\n")
    X, labels_dict = create_images_array(images_dir,
                                         labels=labels_dict,
                                         scale_factor=1.3,
                                         out_size=array_image_dim)
    
    print("-- [DONE] ({} seconds)".format(time.time() - t0))

    # REMOVE UNWANTED LABELS
    for key in ["top", "left", "width", "height"] + bboxes:
        _ = labels_dict.pop(key, 0)

    # MAKE BBOX DATA PROPORTIONAL TO DIMENSIONS OF IMAGE
    # for key in ["bbox", "bbox1", "bbox2", "bbox3", "bbox4", "bbox5"]:
    #     labels_dict[key] = labels_dict[key] / float(array_image_dim)
    labels_dict["bboxes"] = labels_dict["bboxes"] / float(array_image_dim)

    # REPLACE BBOX COORDINATES FOR NULL DIGITS - with a bbox outside the image
    labels_dict = fix_null_digit_bboxes(labels=labels_dict, max_digits=5)

    # --------------------------------------------------------------------------
    #                                                       FIX INCORRECT LABELS
    # --------------------------------------------------------------------------
    #  Train file has a few incorrectly labelled data
    if dataset == "train":
        # i = 26698   VAL: 22827 > 2827
        # np.where((labels["digits"]==[10,2,2,8,2,7]).sum(axis=1) == labels["digits"].shape[1])[0]
        labels_dict["digits"][26698] = [10, 10, 2, 8, 2, 7]
        labels_dict["N"][26698] = 4

        # i=24040     VAL: 03163  > 31637
        # np.where((labels["digits"]==[10,0,3,1,6,3]).sum(axis=1) == labels["digits"].shape[1])[0]
        labels_dict["digits"][24040] = [10, 3, 1, 6, 3, 7]
        labels_dict["N"][24040] = 5

    # --------------------------------------------------------------------------
    #                                                       RESTRICT TO 5 DIGITS
    # --------------------------------------------------------------------------
    print("Restricting to 5 digits", end="")
    # REMOVE ITEMS WITH 6 DIGITS
    sixer_indices = np.where(labels_dict["N"] == 6)[0]  # 29929 ('29930.png')
    for key in labels_dict.keys():
        labels_dict[key] = np.delete(labels_dict[key], sixer_indices, axis=0)
    X = np.delete(X, sixer_indices, axis=0)

    # RESTRICT LABEL ARRAYS TO ONLY TRACK UP TO 5 DIGITS
    for key in ['digits']:
        labels_dict[key] = labels_dict[key][:, -5:]
    print("-- [DONE] ({} seconds)".format(time.time() - t0))

    # --------------------------------------------------------------------------
    #
    # --------------------------------------------------------------------------

    return X, labels_dict


# ==============================================================================
#                                                                 EXPLORING DATA
# ==============================================================================
def explore_data(X, labels):
    print("-" * 60)
    print((" " * 54) + "SHAPES")
    print("-" * 60)
    for key in labels.keys():
        print("{} shape:      {}       TYPE: {}".format(key, labels[key].shape,
                                                        labels[key].dtype))

    print("Images Shape: {}       TYPE: {}".format(X.shape, X.dtype),
          end="\n\n")

    print("-" * 60)
    print((" " * 31) + "DISTRIBUTION OF DIGIT LENGTHS")
    print("-" * 60)
    dist = pd.Series(labels["N"]).value_counts(sort=False, dropna=False,
                                               normalize=False)
    print(dist)


# ==============================================================================
#                                                                    SAVE PICKLE
# ==============================================================================
def save_pickle(obj, f):
    print("Saving: ", f)
    obj2pickle(obj, file=f)
    print("--DONE")


# ==============================================================================
#                                                             PROCESS THE DATA
# ==============================================================================
def process_the_data(data, data_dir, out_dir, limit=None):
    """Cleans the data labels, creates images array of 64x64 images, and
       saves the data to pickle files.

    Args:
        data: (str) "train", "test", "extra"
        data_dir: (str) the base directory where the train, test, and extra
                        direcotories will all be found.
        out_dir: (str) the directory where you want to save the pickle files.
        limit = (None or int) Only proccess a subset of the data
    Returns:

    """
    print("#" * 60)
    print((" " * 47) + "{} DATA".format(data.upper()))
    print("#" * 60)

    # CREATE CLEANED DATA
    mat_file = os.path.join(data_dir, data, "digitStruct.mat")
    images_dir = os.path.join(data_dir, data)
    X, Y = data_cleaning(matfile=mat_file,
                         images_dir=images_dir,
                         dataset=data,
                         limit=limit)

    # SAVE THE DATA
    obj2pickle(Y, file=os.path.join(out_dir, "Y_{}.pickle".format(data)),
               verbose=True)
    obj2pickle(X,
               file=os.path.join(out_dir, "X_{}_cropped64.pickle".format(data)),
               verbose=True)

    # EXPLORATORY PRINTOUT
    explore_data(X=X, labels=Y)


# ==============================================================================
#                                                              MERGE_TRAIN_EXTRA
# ==============================================================================
def merge_train_extra(pickles_dir, shuffle=True):
    """ Merges the train and extra datasets. Optionally shuffles them as well.
        then saves the merged data as two pickle files:
            X_train_extra_cropped64.pickle
            Y_train_extra.pickle

    Args:
        pickles_dir: (str) directory containing the picle files
        shuffle:     (bool) Should it shuffle the data (default is True)
    """
    print("#" * 60)
    print((" " * 34) + "MERGE TRAIN AND EXTRA DATA")
    print("#" * 60)

    # OPEN TRAIN
    X_train = pickle2obj(os.path.join(pickles_dir, "X_train_cropped64.pickle"))
    Y_train = pickle2obj(os.path.join(pickles_dir, "Y_train.pickle"))

    # OPEN EXTRA
    X_extra = pickle2obj(os.path.join(pickles_dir, "X_extra_cropped64.pickle"))
    Y_extra = pickle2obj(os.path.join(pickles_dir, "Y_extra.pickle"))

    # CONCATENATE
    X_merged = np.append(X_train, X_extra, axis=0)
    Y_merged = {}
    for key in Y_train.keys():
        Y_merged[key] = np.append(Y_train[key], Y_extra[key], axis=0)

    # SHUFFLE
    if shuffle:
        random_indices = np.random.permutation(Y_merged["N"].shape[0])
        X_merged = X_merged[random_indices]
        for key in Y_merged.keys():
            Y_merged[key] = Y_merged[key][random_indices]

    # SAVE AS:
    obj2pickle(X_merged,
               file=os.path.join(pickles_dir, "X_train_extra_cropped64.pickle"))
    obj2pickle(Y_merged,
               file=os.path.join(pickles_dir, "Y_train_extra.pickle"))

    # FEEDBACK
    print()
    print("X")
    print("Train Shape : ", X_train.shape)
    print("Extra Shape : ", X_extra.shape)
    print("Merged Shape: ", X_merged.shape)
    print()
    print("Y")
    for key in Y_merged.keys():
        print("{} : {}".format(key.ljust(10, " "), Y_merged[key].shape))


# ==============================================================================
#                                                        INCREASE_REPRESENTATION
# ==============================================================================
def increase_representation(X, labels, min_samples=5000):
    """ Increase the number of samples that come from under-represented digit
        lengths by randomly duplicating items from that group until we have
        a minumum number of samples.

    Args:
        X:            Array of images.
        labels:       dictionary of array labels
        min_samples:  (int) Min number of samples we would like for each length of digits

    Returns:
        X_aug, Y_aug
    """
    num_samples = X.shape[0]
    im_x, im_y = X.shape[1:3]  # 2D Image dimensions

    # Indicies of all samples, grouped by their digit length
    indices_by_len = [np.where(labels["N"] == i + 1)[0] for i in range(5)]

    # CALCULATE HOW BIG THE AUGMENTED ARRAYS WILL BE
    aug_len = sum([max(len(i), min_samples) for i in indices_by_len])

    # INITIALIZE THE AUGMENTED ARRAY OF IMAGES
    X_aug = np.zeros(shape=[aug_len, im_x, im_y], dtype=np.uint8)
    Y_aug = {}
    Y_aug['digits'] = np.empty(shape=[aug_len, 5], dtype=np.uint8)
    Y_aug['N'] = np.empty(shape=aug_len, dtype=np.uint8)
    Y_aug['file'] = np.empty(shape=aug_len, dtype=object)
    Y_aug['dataset'] = np.empty(shape=aug_len, dtype=object)
    Y_aug['bboxes'] = np.empty(shape=[aug_len, 24], dtype=np.float32)

    i_new = -1  # keep track of index in augmented arrays

    # FOR EACH DIGIT LENGTH
    # If there are lengths of digits that contain very little data, then the
    # data is augmented, with multiple random duplicates ensuring that we have
    # a minimum amount of samples for each length of digits.
    for ndigits, indices in enumerate(indices_by_len):
        num_samples = len(indices)
        print("{} digit numbers : {:6.0f}".format(ndigits + 1, num_samples),
              end="")

        if (num_samples < min_samples):
            # Augment the number of indices
            indices = np.random.choice(indices, size=min_samples, replace=True)
            # augment = True
            print(" -> {:5.0f}".format(len(indices)), end="\n")

        else:
            print()

        for i_old in indices:
            # APPEND TO AUGMENTED ARRAYS
            i_new += 1  # The index in the new array
            X_aug[i_new] = X[i_old]
            Y_aug['digits'][i_new] = labels["digits"][i_old]
            Y_aug['N'][i_new] = labels["N"][i_old]
            Y_aug['file'][i_new] = labels["file"][i_old]
            Y_aug['dataset'][i_new] = labels["dataset"][i_old]
            Y_aug['bboxes'][i_new] = labels["bboxes"][i_old]

    assert (aug_len == i_new + 1), \
        "SOMETHING WENT WRONG!: aug_len does not match number " \
        "of new samples created. " \
        \
        #  RANDOM SHUFFLE
    print("Shuffling Array", end="")
    random_indices = np.random.permutation(Y_aug["N"].shape[0])
    X_aug = X_aug[random_indices]
    for key in Y_aug.keys():
        Y_aug[key] = Y_aug[key][random_indices]
    print("--[DONE]")
    print("DONE Increasing Representation")

    return X_aug, Y_aug


# ==============================================================================
#                                           CREATE_INCREASED_REPRESENTATION_DATA
# ==============================================================================
def create_increased_representation_data(pickles_dir):
    print("-" * 60)
    print((" " * 35) + "INCREASING REPRESENTATION")
    print("-" * 60)

    # LOAD DATA
    X = pickle2obj(os.path.join(pickles_dir, "X_train_extra_cropped64.pickle"))
    Y = pickle2obj(os.path.join(pickles_dir, "Y_train_extra.pickle"))

    # INCREASE REPRESENTATION
    X, Y = increase_representation(X, Y, min_samples=5000)

    # SAVE PICKLES
    obj2pickle(X,
               os.path.join(pickles_dir, "X_aug_train_extra_cropped64.pickle"),
               verbose=True)
    obj2pickle(Y,
               os.path.join(pickles_dir, "Y_aug_train_extra.pickle"),
               verbose=True)

    # EXPLORATORY PRINTOUT
    explore_data(X=X, labels=Y)


