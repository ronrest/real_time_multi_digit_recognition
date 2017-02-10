"""
################################################################################
                PROCESSES THE RAW IMAGE FILES AND MAT FILES
################################################################################

Processes the image and mat files taken from the following urls:

- http://ufldl.stanford.edu/housenumbers/train.tar.gz
- http://ufldl.stanford.edu/housenumbers/test.tar.gz
- http://ufldl.stanford.edu/housenumbers/extra.tar.gz

And generates Pickle files that contain numpy arrays of the cropped/resized
images as well as of the labels.

See the README file for details on how to use this script.
################################################################################
"""
from __future__ import print_function, absolute_import

import os
from process_data import merge_train_extra, process_the_data, \
                         create_increased_representation_data

if __name__ == "__main__":
    import argparse
    # --------------------------------------------------------------------------
    #                                                     PROCESS ARGUMENT FLAGS
    # --------------------------------------------------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_dir", default=None, type=str,
                        help="The parent directory containing the directories "
                             "for train, test and extra data. If no argument "
                             "given, then it assumes that the current working "
                             "directory is where the data is.")
    parser.add_argument("-o", "--output_dir", default=None, type=str,
                        help="The directory to output the pickled data, by "
                             "default it places the files in same directory as "
                             "the input data directory")
    parser.add_argument("-d", "--data", default=None, type=str,
                        help="Which dataset to use. Legal options are: \n"
                             "    'train' To process train data. \n"
                             "    'test'  To process test data. \n"
                             "    'extra' To process extra data. \n"
                             "    'merge' To merge the train and extra data. \n"
                             "    'rep'   To create the increased "
                             "            representation data. \n"
                             "     None   To process ALL data. \n"
                             "If no argument provided, it processes ALL the "
                             "data.")
    parser.add_argument("--debug", action='store_true', help="Go into debug mode")

    opts = parser.parse_args()
    # PLACE LIMIT ON NUMBER OF DATA SAMPLES - for debugging purposes
    if opts.debug:
        limit = 1024
        print("#"*70)
        print("NOTE: YOU ARE IN DEBUG MODE")
        print("#" * 70)
    else:
        limit = None
    
    
    #  SET DEFAULT VALUES
    if opts.input_dir is None:
        opts.input_dir = os.path.abspath("")  # Abs path to current working dir
    if opts.output_dir is None:
        opts.output_dir = opts.input_dir  # Output, same as input dir
    
    # --------------------------------------------------------------------------
    #                                    PROCESS THE RELEVANT DATASET TO WORK ON
    # --------------------------------------------------------------------------
    # ALL DATASETS FROM START TO FINISH
    if opts.data is None:
        for dataset in ["extra", "train", "test"]:
            process_the_data(data=dataset,
                             data_dir=opts.input_dir,
                             out_dir=opts.output_dir,
                             limit=limit)
        merge_train_extra(opts.output_dir, shuffle=True)
        create_increased_representation_data(opts.output_dir)
    
    # JUST TRAIN, TEST OR EXTRA
    elif opts.data in ["train", "test", "extra"]:
        process_the_data(data=opts.data,
                         data_dir=opts.input_dir,
                         out_dir=opts.output_dir,
                         limit=limit)
    
    # MERGED DATA
    elif opts.data == "merge":
        merge_train_extra(opts.output_dir, shuffle=True)
    
    # INCREASED REPRESENTATION DATA
    elif opts.data == "rep":
        create_increased_representation_data(opts.output_dir)
    
    # HANDLE WRONG DATASET OPTION
    else:
        assert False, "Incorrect argument for data provided"

