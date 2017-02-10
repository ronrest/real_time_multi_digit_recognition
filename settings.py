import argparse
import os
from file_support.paths import PathsObj


def parse_settings():
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output_name", required=True, type=str, help="Name that will be used to create output directory and zip file")
    parser.add_argument("-m", "--model", required=True, type=str, help="Which model to use (single letter character)")
    parser.add_argument("-i", "--input_dir", default="data", type=str, help="Directory containing the input data")
    parser.add_argument("-e", "--epochs", type=int, default=1, help="Number of epochs")
    parser.add_argument("-a", "--alpha", type=float, default=0.001, help="alpha value")
    parser.add_argument("--conv_dropout", type=float, default=0.1, help="Dropout rate for Conv Layers")
    parser.add_argument("--fc_dropout", type=float, default=0.1, help="Dropout rate for FC Layers")
    
    parser.add_argument("-n", "--data_size", type=int, default=None, help="Number of training samples to use")
    parser.add_argument("-v", "--valid_size", type=int, default=1024, help="Number of validation samples to use")
    parser.add_argument("-b", "--batch_size", type=int, default=128, help="Batch Size")
    
    settings = parser.parse_args()

    settings.image_size = [54, 54]  # Image dimensions to use in model
    settings.image_chanels = 1      # Number of color channels for input images
    # settings.fc_width = 512         # Width of Fully Connected layers
    settings.max_digits = 5         # Number of digits to recognize
    
    return settings


def establish_paths(output_name, input):
    """
    output_name: output name (will be converted to a path where things will be saved)
    input: directory where the data is stored
    """

    # Short path function names
    abspath = os.path.abspath
    dirname = os.path.dirname
    basename = os.path.basename
    
    paths = PathsObj()
    # WORKING DIR
    paths.add("script_file", abspath(__file__))
    paths.add("working_dir", os.getcwd()) #dirname(paths.script_file))
    paths.add("working_dir_name", basename(paths.working_dir))

    # INPUT PATHS
    paths.add("data_dir", input) # "/home/ronny/TEMP/house_numbers_SVHN/format1"
    paths.add("X_train", "X_aug_train_extra_cropped64.pickle", root="data_dir")
    paths.add("Y_train", "Y_aug_train_extra.pickle", root="data_dir")
    paths.add("X_test", "X_test_cropped64.pickle", root="data_dir")
    paths.add("Y_test", "Y_test.pickle", root="data_dir")

    # OUTPUT PATHS
    paths.add("train_results", "results", root="working_dir", create_dir=True)
    paths.add("output_name", output_name)
    paths.add("output", output_name, root="train_results", create_dir=True)

    # CHECKPOINTS
    paths.add("checkpoint_dir", "checkpoints", root="output", create_dir=True)
    paths.add("checkpoint", "checkpoint.chk", root="checkpoint_dir")
    paths.add("checkpoint_max", "checkpoint_max.chk", root="checkpoint_dir")
    
    # EVALS
    paths.add("evals_dir", "evals", root="output", create_dir=True)
    paths.add("evals", "evals.pickle", root="evals_dir")
    paths.add("evals_max", "evals_max.pickle", root="evals_dir")
    paths.add("learning_curves", "learning_curves.png", root="evals_dir")

    # SETTINGS/OPTS
    paths.add("settings_text_file", "settings.txt", root="output")
    paths.add("settings_pickle_file", "settings.pickle", root="output")

    # COMPARISONS - Where files to compare different models is stored
    paths.add("model_comparisons_dir", "comparisons", "train_results", create_dir=True)
    
    # TENSORBOARD
    paths.add("tensorboard_dir", "tensorboard", root="output", create_dir=True)

    # VISUALISATIONS
    paths.add("epoch_vis", "epoch_vis", root="output", create_dir=True)

    return paths




