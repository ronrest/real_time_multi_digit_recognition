"""
################################################################################
                                                                     DESCRIPTION
################################################################################
Saves several images that explore the data that will be used to train the model

################################################################################
"""
from __future__ import print_function

from main import np
from main import os

import pandas as pd
from matplotlib import pyplot as plt

from process_data.helpers import mat_data_to_dict_of_arrays
from support import verbose_print, verbose_print_done, print_headers

from vis import grid_of_sample_images
from image_processing import load_image, pil2array, ANTIALIAS
from support import array_of_digit_arrays_to_str
from PIL import Image

from file_support import pickle2obj


################################################################################
#                                                                       SETTINGS
################################################################################
verbose = True
data_dir = "data"

################################################################################
#                                                                           DATA
################################################################################
print_headers("EXTRACTING LABELS - NOTE: this may take a LONG time", verbose=verbose)

verbose_print("Extracting labels from TRAIN mat file", verbose, end="")
matfile = os.path.join(data_dir, "train", "digitStruct.mat")
Y_train = mat_data_to_dict_of_arrays(matfile, limit=None)
verbose_print_done(verbose)

verbose_print("Extracting labels from EXTRA mat file", verbose, end="")
matfile = os.path.join(data_dir, "extra", "digitStruct.mat")
Y_extra = mat_data_to_dict_of_arrays(matfile, limit=None)
verbose_print_done(verbose)

verbose_print("Extracting labels from TEST mat file", verbose, end="")
matfile = os.path.join(data_dir, "test", "digitStruct.mat")
Y_test = mat_data_to_dict_of_arrays(matfile, limit=None)
verbose_print_done(verbose)


Y = {"train": Y_train,
     "extra": Y_extra,
     "test": Y_test
     }

################################################################################
#                                                                        EXPLORE
################################################################################

# ------------------------------------------------------------------------------
#                                                                  DATASET SIZES
# ------------------------------------------------------------------------------
datasets = ["train", "extra", "test"]
print("DATASET SIZES")
for dataset in datasets:
    print(dataset+" : ", Y[dataset]["N"].shape[0])


# ------------------------------------------------------------------------------
#                                              VISUALIZE SAMPLES FROM TRAIN DATA
# ------------------------------------------------------------------------------
# LOAD IMAGES INTO AN ARRAY
n_samples = 16
dimx, dimy = 50,50
imgs = np.empty(shape=[n_samples, dimx, dimy, 3])
for i in range(n_samples):
    file = Y["train"]["file"][i]
    img = load_image(os.path.join(data_dir, "train", file))
    img.thumbnail([dimx,dimy], ANTIALIAS)
    x = (dimx-img.width)/2
    y = (dimy-img.height)/2
    img_box = Image.new('RGB', (dimx,dimy), (255, 255, 255))
    img_box.paste(img, (x, y))
    imgs[i] = pil2array(img_box)
    # img_box.show()

# VIEW LABELLED IMAGES AS A GRID
grid_of_sample_images(imgs,
                      # labels=fivers_labels ,
                      gridsize=(4,4),
                      label_color="#000000",
                      title="Sample of Images from Train Dataset",
                      saveto="imgs/raw_train_sample.png")



# ------------------------------------------------------------------------------
#                                          VISUALIZE JUST THE FIVE DIGIT NUMBERS
# ------------------------------------------------------------------------------
i_fivers = np.where(Y["train"]["N"] == 5)[0]
n_fivers = i_fivers.shape[0]
fivers_labels = Y["train"]["digits"][i_fivers]
fivers_labels = array_of_digit_arrays_to_str(fivers_labels, null=10)

# LOAD IMAGES INTO AN ARRAY
imgs = np.empty(shape=[n_fivers, 100,100])
for i, file in enumerate(Y["train"]["file"][i_fivers]):
    img = load_image(os.path.join(data_dir, "train", file))
    img = img.convert('L') # Convert to greyscale
    img = img.resize([100,100], ANTIALIAS)
    imgs[i] = pil2array(img)

# VIEW LABELLED IMAGES AS A GRID
grid_of_sample_images(imgs,
                      labels=fivers_labels ,
                      gridsize=(3,3),
                      label_color="#000000",
                      title="5 Digit Numbers in Train Dataset",
                      saveto="imgs/five_digit_numbers_train.png")



# ------------------------------------------------------------------------------
#                                             PLOT DISTRIBUTION OF DIGIT LENGTHS
# ------------------------------------------------------------------------------
fig, ax = plt.subplots(1, 3, figsize=(10,4))
fig.suptitle('Distribution of digit lengths for each dataset', y=1.0, fontsize=14)

for j, dataset in enumerate(datasets):
    # CREATE A TALLY OF DIGIT LENGTHS
    dist = pd.Series(Y[dataset]["N"]).value_counts(sort=False, dropna=False, normalize=False)
    x = dist.axes[0]
    y = np.array(dist)
    
    # PLOT AS A BAR GRAPH - Using log scale for y axis
    ax[j].bar(x, y, width=0.95, alpha=0.7, color="#307EC7", align="center",lw=0)
    ax[j].set_yscale('log')
    ax[j].set_title("{} Dataset".format(dataset.title()), y=1.0,  size=12)
    ax[j].spines['top'].set_color('none')
    ax[j].spines['right'].set_color('none')
    ax[j].xaxis.set_ticks_position('bottom')
    ax[j].yaxis.set_ticks_position('left')
    
    # ANNOTATE THE PLOT
    for i in range(len(x)):
        ax[j].annotate(y[i],
                       size=10,
                       xy=(x[i], y[i]), # Position of the corresponding bar
                       xytext=(0, 2),  # Offset text
                       textcoords='offset points', # Use offset points
                       ha='center',     # Horizontal alignment
                       va='center')     # Vertical alignment
    fig.tight_layout()
    
# SAVE THE PLOT
fig.savefig("imgs/raw_digit_distributions.png")



# ------------------------------------------------------------------------------
#                                                    MERGED TRAIN AND EXTRA DATA
# ------------------------------------------------------------------------------
Y_train_extra = pickle2obj(os.path.join(data_dir, "Y_train_extra.pickle"), verbose=verbose)
print("MERGED TRAIN-EXTRA DATA samples: ", Y_train_extra["N"].shape[0])


# ------------------------------------------------------------------------------
#                             DISTRIBUTION OF EACH DIGIT FOR EACH DIGIT POSITION
# ------------------------------------------------------------------------------
def myFunc(s):
    return s.value_counts(sort=False, dropna = False)

df = pd.DataFrame(Y_train_extra["digits"]).apply(myFunc , axis=0)
df.columns = [1,2,3,4,5]

fig, axes = plt.subplots(2, 3, figsize=(8, 6), sharex=False, sharey=True)
fig.suptitle('Distribution of digits at each digit position', y=1.0, fontsize=14)
plt.ylim([0.5,1e6])
df[1].plot(kind="bar", logy=True, ax=axes[0,0], use_index=True, title="Position 1", width=0.8, alpha=0.7, color="#307EC7")
df[2].plot(kind="bar", logy=True, ax=axes[0,1], title="Position 2", width=0.8, alpha=0.7, color="#307EC7")
df[3].plot(kind="bar", logy=True, ax=axes[0,2], title="Position 3", width=0.8, alpha=0.7, color="#307EC7")
df[4].plot(kind="bar", logy=True, ax=axes[1,0], title="Position 4", width=0.8, alpha=0.7, color="#307EC7")
df[5].plot(kind="bar", logy=True, ax=axes[1,1], title="Position 5", width=0.8, alpha=0.7, color="#307EC7")
axes[1,2].get_yaxis().set_visible(False)  # Remove axis ticks
axes[1,2].get_xaxis().set_visible(False)  # Remove axis ticks
axes[1,2].spines['top'].set_color('none')    # Remove Top Border
axes[1,2].spines['right'].set_color('none')  # Remove Right Border
axes[1,2].spines['bottom'].set_color('none')    # Remove Top Border
axes[1,2].spines['left'].set_color('none')  # Remove Right Border
fig.savefig("imgs/digit_position_distributions.png")


#            1       2       3      4        5
# 0        NaN      32    2793  16792  30880.0
# 1       81.0    9866   41129  31307  22037.0
# 2       31.0    3143   27567  30330  24253.0
# 3        7.0    1172   17712  27091  23279.0
# 4        1.0     762   12665  23938  20724.0
# 5        1.0     400    9538  22204  28227.0
# 6        1.0     228    6619  18540  21921.0
# 7        1.0     161    6077  19967  23387.0
# 8        NaN      69    4008  15956  20369.0
# 9        NaN      63    3268  15107  20677.0
# 10  235631.0  219858  104378  14522      NaN

