from main import np
from main import os

from matplotlib import pyplot as plt
from image_processing import draw_boundingboxes
from image_processing import array2pil, pil2array
import copy

from support import array_of_digit_arrays_to_ints
from file_support import maybe_mkdir

from evals import batch_iou, batch_multi_column_iou


#  =============================================================================
#                                                                        GLOBALS
#  =============================================================================
GREEN = "#52F900"
RED = "#FF0D02"
BLUE = "#0060F9"
ORANGE = "#F98E00"
DARK_GREEN = "#417D08"
DARK_RED = "#A80A00"

NICE_GREEN = "#73AD21"
NICE_BLUE = "#307EC7"
GUAVA = "#FF4F40"


# ==============================================================================
#                                                           PLOT_TRAINING_CURVES
# ==============================================================================
def plot_training_curves(evals, crop=(None, None), saveto=None):
    """ Plots training curves given a dictionary-like object with lists for the
        the following keys:
        - "pda_train"
        - "pda_valid"
        - "wna_train"
        - "wna_valid"
        - "iou_train"
        - "iou_valid"
        - "alpha"
        - "loss"
    """
    low = crop[0]
    high = crop[1]
    
    if saveto:
        plt.ioff()  # prevent figure from displaying
    
    fig, axes = plt.subplots(2, 2)
    fig.suptitle('Evaluation per epoch of training', fontsize=15)
    
    # WNA
    axes[0, 0].plot(evals["wna_train"][low:high], color=GUAVA, label="WNA train")
    axes[0, 0].plot(evals["wna_valid"][low:high], color=NICE_GREEN, label="WNA valid")
    
    showmax = True
    if showmax:
        i_max = np.array(evals["wna_valid"][low:high]).argmax()
        axes[0,0].axvline(x=i_max, color=NICE_BLUE, ls="solid", label="Best")

    axes[0, 0].legend(loc="lower right", frameon=False)
    axes[0, 0].set_title("WNA")

    # PDA
    axes[1, 0].plot(evals["pda_train"][low:high], color=GUAVA, label="PDA train")
    axes[1, 0].plot(evals["pda_valid"][low:high], color=NICE_GREEN, label="PDA valid")
    axes[1, 0].legend(loc="lower right", frameon=False)
    axes[1, 0].set_title("PDA")
    
    # LOSSES
    axes[1, 1].plot(evals["loss"][low:high], color=GUAVA)
    axes[1, 1].set_title("Loss")
    
    # IOUs
    axes[0, 1].plot(evals["iou_train"][low:high], color=GUAVA, label="train")
    axes[0, 1].plot(evals["iou_valid"][low:high], color=NICE_GREEN, label="valid")
    axes[0, 1].legend(loc="lower right", frameon=False)
    axes[0, 1].set_title("IoU")

    # # ALPHAS
    # axes[1, 0].plot(evals["alpha"][low:high], color=NICE_BLUE)
    # axes[1, 0].set_title("Alpha")
    
    
    if saveto:
        parent_dir = os.path.abspath(os.path.join(saveto, os.pardir))
        if not os.path.exists(parent_dir):
            os.makedirs(parent_dir)
        fig.savefig(saveto)
        plt.close(fig)  # needed to prevent image from displaying
    else:
        fig.show()


# ==============================================================================
#                                                                  ARRAY_TO_PLOT
# ==============================================================================
def array_to_plot(a, reshape=None, ax=None, cmap=None, vrange=(0,255), ticks=False):
    """array to image
    Takes a numpy array that contains pixel information, and displays the image
    as a matlotlib plot. You can optionally pass a matplot lib axis and it will
    polulate that axis with the image instead.

    Args:
        a:       numpy array containing a single image.
        reshape: (None or tuple) (default=None)
                 specify a tuple of two ints (width, height) if
                 the input array is not already that shape.
        ax:      (matplotlib axis, or None) (default=None)
                 If you want the plot to be placed
                 inside an existing figure, then specify the axis to place
                 the new image into, otherwise a value of `None` will create a
                 new figure.
        cmap:    (default = None)
                 colormap. eg "gray" for grayscale. None for default colormap of
                 matplotlib's `imshow()`.
        vrange:  (tuple of two numbers) Range of values over which pixels
                 COULD take. eg [0,255], or [0,1]
        
        ticks:   (boolean) (default = False)
                 Should it show the x and y axis tick marks?

    Returns:
        If `ax` was specified, then it returns the `ax` with the image.
        Otherwise it doesnt return anything, and just shows the image.
    """
    vmin, vmax = vrange  # Range of values that pixel data comes from
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(1, 1))
    if reshape is None:
        array = a
    else:
        array = a.reshape(
            reshape)  # COnvert a 1D array to a 60x40 2D array.
    
    ax.imshow(array, cmap=cmap, vmin=vmin, vmax=vmax)  # , interpolation='bicubic')
    
    # Hide the ticks and tick labels
    if not ticks:
        ax.get_yaxis().set_visible(False)
        ax.get_xaxis().set_visible(False)
    if ax is not None:
        return ax
    else:
        fig.show()


# ==============================================================================
#                                                   OVERLAY_BOUNDING_BOXES_ARRAY
# ==============================================================================
def overlay_bounding_boxes_array(a, bboxes,
                                 outline="#FF0000FF",
                                 fill=None,
                                 proportional=True):
    """ Takes an array `a` of images, and bounding box information,
        (optionally a second set of bounding boxes if you want to
        compare true and predicted bounding boxes).

        It returns a copy of the array, but with the bounding boxes
        overlayed on the pixel information.

        NOTE: that it converts to RGB, so it returns a 4D array with
        3 channels on the final layer.

    Args:
        a:      (array) array of images
        bboxes: (array) array, where each row contains the bounding boxes for
                each image. The shape is a mulitple of 4, with each group of
                four columns being a single bounding box: [x1,y1,x2,y2]
        outline: (str) Color for bounding box outline  (accepts RGB or RGBA)
        fill:   (str) Color for bounding box fill (accepts RGB or RGBA)
        proportional: (bool) True if the bbox values are proportional to the
                image dimensions (values between 0-1)
                Set to False if the bbox values are absolute pixel coordinates.

    Returns:
        Array with the following dimensions:
            [n_samples, dim_x, dim_y, 3]
    """
    n_samples, x, y = a.shape[:3]
    out = np.empty(shape=[n_samples, x, y, 3], dtype=np.uint8)

    for i in range(n_samples):
        im = array2pil(a[i], mode=None)
        im = draw_boundingboxes(im,
                                bboxes=bboxes[i],
                                to_rgb=True,
                                outline=outline,
                                fill=fill,
                                proportional=proportional)
        out[i] = pil2array(im)
    return out



# ==============================================================================
#                                                          GRID_OF_SAMPLE_IMAGES
# ==============================================================================
def grid_of_sample_images(a, labels=None, labels2=None, gridsize=(4,4),
                          reshape=None,
                          cmap="gray", saveto=None, show=True, random=False,
                          vrange=(0,255), title="", label_font_size=9,
                          label_color=DARK_GREEN,
                          label2_color=DARK_RED,
                          seed=None):
    """
    Takes an array of image data, where the first axis represents the number
    of samples. eg [num samples, image]
    Args:
        a:          The array of images
        labels:     (list of strings) label for each image.
        gridsize:   (tuple of two ints)
                    Specify the grid size (num columns, num rows)
        reshape:    The dimensions to reshape the image to (widht, height)
        cmap:       Colormap. "gray" or None.
        saveto:     (optional) (default=None)
                    file to save the output image to.
        show:       (bool) Show image on screen?
        random:     randomly sample?
        vrange:     (list of two numbers) Range of values over which pixels
                    COULD take. eg [0,255], or [0,1]
        title:      (string) Title for this diagram.
        label_font_size: (int) font size for the labels
        seed:       (int) seed for random generator

    Returns:
        None, it plots the image.
    """
    np.random.seed(seed=seed)

    if not show:
        plt.ioff()

    # SETTINGS
    n = gridsize[0]*gridsize[1]     # Number of images to sample
    indices = np.arange(a.shape[0]) #
    im_shape = a.shape[1:]          # 2D shape of images
    
    # SAMPLE N IMAGES
    if random:
        indices = np.random.choice(indices, size=n, replace=False)
    images = a[indices][:n]
    actual_n_images = images.shape[0]   # Number of actual images in array
    labels = None if labels is None else np.array(labels)[indices][:n]
    labels2 = None if labels2 is None else np.array(labels2)[indices][:n]

    # HANDLE GRAYSCALE IMAGES with a chanels axis
    if im_shape[-1] == 1:
        images = images.reshape(n, im_shape[0], im_shape[1])

    # PLOT
    n_side = int(n ** (1 / 2.))  # Number of images per row/col
    fig, axes = plt.subplots(gridsize[1], gridsize[0])  # figsize=(10,10)
    axes = np.array(axes).flatten() # Unroll axes to a flat list
    # axes = [item for row in axes for item in row]
    fig.suptitle(title, fontsize=15,
                 fontdict={"fontweight": "extra bold"})
    
    # PLOT EACH IMAGE
    for i, ax in enumerate(axes):
        if (i+1) > actual_n_images:
            # FILL WITH BLANK IMAGES IF NOT ENOUGH IMAGES TO FIT GRID
            ax.get_yaxis().set_visible(False)
            ax.get_xaxis().set_visible(False)
            ax.set_aspect('equal')
            
        else:
            ax = array_to_plot(images[i], reshape=reshape, cmap=cmap,
                                vrange=vrange, ax=ax)
            ax.set_aspect('equal')
            
            # CELL LABEL
            if labels is not None and labels2 is not None:
                pos2 = -(label_font_size/2)
                pos1 = pos2-(1.2*label_font_size)
                ax.text(0, pos1, labels[i], fontsize=label_font_size, color=label_color, ha='left')
                ax.text(0, pos2, labels2[i], fontsize=label_font_size, color=label2_color,ha='left')
                fig.subplots_adjust(wspace=0.01, hspace=0.4)
            elif labels is not None:
                ax.set_title(labels[i], color=label_color, fontsize=label_font_size)
                fig.subplots_adjust(wspace=0.01, hspace=0.3)
            elif labels2 is not None:
                ax.set_title(labels2[i], color=label2_color, fontsize=label_font_size)
                fig.subplots_adjust(wspace=0.01, hspace=0.3)


    if saveto:
        fig.savefig(saveto)
    if show:
        # plt.show()
        fig.show()
    else:
        plt.close(fig)


# ==============================================================================
#                                                          GRID_OF_SAMPLE_BBOXES
# ==============================================================================
def grid_of_sample_bboxes(a, bboxes, bboxes2=None, gridsize = (5,5),
                          fill=None, outline=GREEN,
                          fill2=None, outline2=RED,
                          proportional=True, **kwargs):
    """ Takes an array of images, and an array of bounding boxes and draws a
        grid of those images with the bounding boxes drawn on.

        Optionally takes a second array of bounding boxes, and draws them in a
        different color, eg to compare predicted vs true bounding boxes.

    Args:
        a:              (numpy array) Images array, in one of the following two
                        shapes:
                        - [n_samples, width, height]
                        - [n_samples, width, height, n_chanels]
        bboxes:         (numpy array) bounding boxes for each image.
                        - [n_samples, 4*num_bounding_boxes]
                        Where each group of 4 columns represets the bbox as:
                        - [x1, y1, x2, y2]
        bboxes2:        (numpy array) Optional second set of bounding boxes.
        fill:           (str) fill color for first bbox
        outline:        (str) outline color for first bbox
        fill2:          (str) fill color for second bbox
        outline2:       (str) outline color for second bbox
        proportional:   (bool) Whether the bounding box data is proportional
                        to the image dimensions (ie, in the range 0-1)
        **kwargs:       Additional key word arguments to pass on to
                        `grid_of_sample_images()`
    """
    # LIMIT THE SIZE OF DATA
    n_samples = gridsize[0]*gridsize[1]
    X = copy.deepcopy(a[:n_samples])
    bboxes = bboxes[:n_samples]
    if bboxes2 is not None:
        bboxes2 = bboxes2[:n_samples]
    
    # DRAW BOUNDING BOXES ON IMAGES
    X = overlay_bounding_boxes_array(X, bboxes=bboxes,
                                     proportional=proportional,
                                     outline=outline, fill=fill)
    if bboxes2 is not None:
        X = overlay_bounding_boxes_array(X, bboxes=bboxes2,
                                         proportional=proportional,
                                         outline=outline2, fill=fill2)
    # PLOT THE SAMPLE IMAGES
    grid_of_sample_images(X, gridsize=gridsize, **kwargs)


# ==============================================================================
#                                                           EPOCH_VISUALISATIONS
# ==============================================================================
def epoch_visualisations(path, epoch, data, bboxes_pred, digits_pred):
    """ Set of visualisations to be drawn and saved at the end of each epoch.
    
    Args:
        path:   (str) path to the directory that will hold all the epoch
                visualisations
        epoch:  (int) epoch number
        data:   (DataObj) A DataObj object, containing the attributes:
                - Y
                - bboxes
        Y:      (dict) labels dictionary for the sample data
        bboxes_pred: (numpy array) predicted bboxes for the sample images
        digits_pred: (numpy array) predicted digits for the sample images
    """
    # ESTABLISH PATHS
    whole_bbox_dir = os.path.join(path, "whole_bbox")
    digit_bbox_dir = os.path.join(path, "digit_bbox")
    digits_pred_dir = os.path.join(path, "digit_preds")
    dificult_digits_dir = os.path.join(path, "difficult_digits")
    maybe_mkdir(whole_bbox_dir)
    maybe_mkdir(digit_bbox_dir)
    maybe_mkdir(digits_pred_dir)
    maybe_mkdir(dificult_digits_dir)
    file_name = "{}.png".format(str(epoch).zfill(4))
        
    # WHOLE BOUNDING BOXES
    grid_of_sample_bboxes(data.X[:25],
                          bboxes=data.whole_bboxes[:25],
                          bboxes2=bboxes_pred[:25, :4],
                          labels=batch_iou(bboxes_pred[:25, :4], data.whole_bboxes[:25]),
                          gridsize=[5, 5],
                          label_font_size=8,
                          saveto=os.path.join(whole_bbox_dir, file_name),
                          show=False,
                          outline=GREEN + "BB",
                          fill=GREEN + "22",
                          outline2=RED,
                          fill2=None
                          )
    
    # DIGIT BOUNDING BOXES
    grid_of_sample_bboxes(data.X[:25],
                          bboxes=data.digit_bboxes[:25],
                          bboxes2=bboxes_pred[:25, 4:],
                          labels=batch_multi_column_iou(bboxes_pred[:25, 4:],
                                                      data.digit_bboxes[:25]).mean(axis=1),
                          gridsize=[5, 5],
                          label_font_size=8,
                          saveto=os.path.join(digit_bbox_dir, file_name),
                          show=False,
                          outline=GREEN + "BB",
                          fill=GREEN + "22",
                          outline2=RED,
                          fill2=None
                          )
    
    # DIGIT PREDICTIONS
    labels = array_of_digit_arrays_to_ints(data.Y[:25], null=10)
    labels_pred = array_of_digit_arrays_to_ints(digits_pred[:25], null=10)
    
    grid_of_sample_images(data.X[:25],
                          labels=labels,
                          labels2=labels_pred,
                          gridsize=[5, 5],
                          label_font_size=8,
                          saveto=os.path.join(digits_pred_dir, file_name),
                          show=False
                          )

    # VISUALISE DIFICULT CASES
    dificult_indices = (digits_pred != data.Y).any(axis=1)
    dificult_data = data.extract_items(indices=dificult_indices, deepcopy=True)
    dificult_preds = digits_pred[dificult_indices]
    labels = array_of_digit_arrays_to_ints(dificult_data.Y[:36], null=10)
    labels_pred = array_of_digit_arrays_to_ints(dificult_preds[:36], null=10)

    grid_of_sample_images(dificult_data.X[:36],
                          labels=labels,
                          labels2=labels_pred,
                          gridsize=[6, 6],
                          label_font_size=7,
                          saveto=os.path.join(dificult_digits_dir, file_name),
                          show=False
                          )



