"""
################################################################################
#                                                               IMAGE_PROCESSING
################################################################################
                Convenience functions for processing images.
"""
from PIL import ImageEnhance, Image, ImageFilter, ImageDraw
from main import NULL_BBOX
from main import np
from main import copy
from support import repeat_array
randint = np.random.randint

# GLOBAL VARIABLES
MODE_CHANELS = {"RGBA": 4, "RGB": 3, "L": 1} # num channels for PIL image modes
ANTIALIAS = Image.ANTIALIAS
BILINEAR = Image.BILINEAR

# ==============================================================================
#                                                                     LOAD IMAGE
# ==============================================================================
def load_image(f):
    return Image.open(f)

# ==============================================================================
#                                                                      PIL2ARRAY
# ==============================================================================
def pil2array(im, dtype=np.uint8):
    """From PIL image to numpy array"""
    return np.asarray(im, dtype=dtype)


# ==============================================================================
#                                                                      ARRAY2PIL
# ==============================================================================
def array2pil(a, mode="grey"):
    """From numpy array to PIL image"""
    if mode is not None and mode.lower() in ["grey", "gray", "l"]:
        mode = "L"
    return Image.fromarray(a, mode=mode)


# ==============================================================================
#                                                             DRAW_BOUNDINGBOXES
# ==============================================================================
def draw_boundingboxes(im, bboxes,
                       to_rgb=True,
                       outline="#FF000099",
                       fill=None,
                       proportional=False):
    """ Takes a PIL image, and an array of bounding boxes, and returns a new
        image with those bounding boxes drawn on.

    Args:
        im: (PIL image)
        bboxes:     (array-like)
                    an array containing one or more bounding boxes to draw.
                    The shape can either be:
        
                    - 1D array:  [num_bboxes*4]
                    - 2D array:  [num_bboxes, 4]
        
                    where each individual bounding box is composed of the
                    following 4 columns:
        
                    - [x1,y1,x2,y2]
        outline:    (str) color of bounding box outline (accepts RGB, and RGBA).
        fill:       (str) color of bounding box fill (accepts RGB, and RGBA).
        
        proportional: (bool)
                    Is the bounding boxes coordinates set as a
                    proportion of the image dimensions? eg, values between 0-1
    Returns:
        A new image with bounding box drawn in the color specified.
    """
    im2 = im.copy()
    
    # RESHAPE BBOXES TO HAVE ONE BBOX PER ROW
    bboxes = np.array(bboxes)
    num_bboxes = bboxes.size // 4
    bboxes = bboxes.reshape(num_bboxes, 4)
    
    # TRANSFORM IMAGE TO RGB
    if to_rgb:
        im2 = im2.convert('RGB')
    
    # HANDLE PROPORTIONAL BOUNDING BOX COORDINATES
    if proportional:
        imdims = np.tile(im.size, 2)
        bboxes = np.multiply(imdims, bboxes).astype(np.int32)
    
    # DRAW THE BOUNDING BOX
    draw = ImageDraw.Draw(im2, mode="RGBA")
    for bbox in bboxes:
        draw.rectangle(list(bbox), fill=fill, outline=outline)
    return im2


# ==============================================================================
#                                                                    RANDOM_CROP
# ==============================================================================
def random_crop(im, crop_size=None):
    """
    Assumes that both im, and the output will be square images.

    Args:
        im:         PIL image
        crop_size:   (int or None) the dimension along one edge of a square

    Returns:
        PIL image of size crop_size, randomly cropped from `im`.
        If crop_size = None, it returns the original image.
    """
    if crop_size is None:
        return im
    else:
        max_offset = im.size[0] - crop_size
        x_offset = randint(0, max_offset + 1)
        y_offset = randint(0, max_offset + 1)
        im2 = im.crop((x_offset, y_offset,
                       x_offset + crop_size,
                       y_offset + crop_size))
        return im2


# ==============================================================================
#                                                              RANDOM_BRIGHTNESS
# ==============================================================================
def random_brightness(im, sd=0.5, min=0.2, max=20):
    """Creates a new image which randomly adjusts the brightness of `im` by
       randomly sampling a brightness value centered at 1, with a standard
       deviation of `sd` from a normal distribution. Clips values to a
       desired min and max range.

    Args:
        im:   PIL image
        sd:   (float) Standard deviation used for sampling brightness value.
        min:  (int or float) Clip brightnes value to be no lower than this.
        max:  (int or float) Clip brightness value to be no higher than this.


    Returns:
        PIL image with brightness randomly adjusted.
    """
    
    brightness = np.clip(np.random.normal(loc=1, scale=sd), min, max)
    enhancer = ImageEnhance.Brightness(im)
    return enhancer.enhance(brightness)


# ==============================================================================
#                                                                RANDOM_CONTRAST
# ==============================================================================
def random_contrast(im, sd=0.5, min=0, max=10):
    """Creates a new image which randomly adjusts the contrast of `im` by
       randomly sampling a contrast value centered at 1, with a standard
       deviation of `sd` from a normal distribution. Clips values to a
       desired min and max range.

    Args:
        im:   PIL image
        sd:   (float) Standard deviation used for sampling contrast value.
        min:  (int or float) Clip contrast value to be no lower than this.
        max:  (int or float) Clip contrast value to be no higher than this.

    Returns:
        PIL image with contrast randomly adjusted.
    """
    contrast = np.clip(np.random.normal(loc=1, scale=sd), min, max)
    enhancer = ImageEnhance.Contrast(im)
    return enhancer.enhance(contrast)


# ==============================================================================
#                                                                    RANDOM_BLUR
# ==============================================================================
def random_blur(im, min=0, max=5):
    """ Creates a new image which applies a random amount of Gaussian Blur, with
        a blur radius that is randomly chosen to be in the range [min, max]
        inclusive.

    Args:
        im:   PIL image
        min:  (int) Min amount of blur desired.
        max:  (int) Max amount of blur desired.


    Returns:
        PIL image with random amount of blur applied.
    """
    blur_radius = randint(min, max + 1)
    if blur_radius == 0:
        return im
    else:
        return im.filter(ImageFilter.GaussianBlur(radius=blur_radius))


# ==============================================================================
#                                                                   RANDOM_NOISE
# ==============================================================================
def random_noise(im, sd=5):
    """Creates a new image which has random noise.
       The intensity of the noise is determined by first randomly choosing the
       standard deviation of the noise as a value between 0 to `sd`.
       This value is then used as the standard deviation for randomly sampling
       individual pixel noise from a normal distribution.
       This random noise is added to the original image pixel values, and
       clipped to keep all values between 0-255.

    Args:
        im:   PIL image
        sd:   (int) Max Standard Deviation to select from.

    Returns:
        PIL image with random noise added.
    """
    # TODO: Maybe make the alpha chanel NOT have any noise, or set as optional
    noise_sd = np.random.randint(0, sd)
    
    if noise_sd > 0:
        im2 = np.asarray(im, dtype=np.uint8)
        noise = np.random.normal(loc=0, scale=noise_sd, size=im2.shape)
        
        # clipping to keep values between [0, 255]
        im2 = np.clip(im + noise, 0, 255).astype(np.uint8)
        return array2pil(im2, mode=im.mode)
    else:
        return im


# ==============================================================================
#                                                                RANDOM_ROTATION
# ==============================================================================
def random_rotation(im, max=10, expand=True, original_dims=True):
    """ Creates a new image which is rotated by a random amount between
        [-max, +max] inclusive.

    Args:
        im:     PIL image
        max:    (int) Max angle (in either direction).
        expand: (bool) expand image to ensure that none of the content is
                cropped out of the frame.
        oroginal_dims: (bool) resize the image back to its original size if
                the image becomes larger as a result of expanding?

    Returns:
        PIL image with random rotation applied.
    """
    angle = randint(-max, max + 1)
    if angle == 0:
        return im
    else:
        im2 = im.rotate(angle, resample=Image.BILINEAR, expand=expand)
        
        # Rescale back to original dimensions
        if expand and original_dims:
            im2 = im2.resize(im.size, Image.ANTIALIAS)
        return im2




# ==============================================================================
#                                                                  LABELLED_CROP
# ==============================================================================
def labelled_crop(im, bboxes, N, min_crop=54, outsize=[54, 54], random=False):
    """ Takes an image, and bounding boxes array of the following shape:
            [24,]
        Where the first four columns represent the bounding box for the entire
        number in the form:
            [x1, y1, x2, y2]
        and each subsequent group of four columns represent the bboxes for
        each individual digit.

        It crops the image provided, and updates the bounding box details.

        If `random` = False, then it takes a centered crop, of size:
            [min_crop, min_crop]
        Otherwise, it randomnly picks a crop size between the existing image
        size, and `min_crop`. It also randomly choses an offset fot he
        the cropping region.

        It resizes the cropped image to the size specified by `outsize`.

    Args:
        im:         (PIL image)
        bboxes:     (numpy array)
        N:          (int) number of digits the number contains.
                    This will be used for fixing bounding box data for
                    null digits.
        min_crop:   (int) minimum size of the crop (assuming square crop.
        outsize:    (tuple of 2 ints) Dimensions you want image to be resized to
        random:     (bool) Should it take a random sized (and positioned) crop?

    Returns:
        Tuple of the following two things:
            - PIL image (of the cropped image)
            - numpy array of the update bounding box coordiates.
    """
    # DIMS
    img_size = im.size[0]
    num_bboxes = 6
    
    # GENERATE A RANDOM CROP SIZE ALONG EACH AXIS
    if random:
        crop_size = np.random.randint(low=min_crop, high=img_size + 1, size=2)
    else:
        crop_size = np.array([min_crop, min_crop])
    
    # CALCULATE OFFSETS FOR CROP
    if random:
        offset_x = np.random.randint(low=0, high=(img_size - crop_size[0]) + 1)
        offset_y = np.random.randint(low=0, high=(img_size - crop_size[1]) + 1)
    else:
        offset_x = offset_y = int((img_size - min_crop) / 2.0)
    
    offsets = repeat_array(np.array([offset_x, offset_y]), num_bboxes * 2)
    
    # Offset proportional to the original image
    offsets_prop = offsets / float(img_size)
    
    # CROP THE IMAGE
    crop_region = np.array([offsets[0],
                            offsets[1],
                            offsets[2] + crop_size[0],
                            offsets[3] + crop_size[1]]
                           )
    im2 = im.crop(crop_region)
    im2 = im2.resize(size=outsize, resample=ANTIALIAS)
    
    # SHIFT AND SCALE BBOX COORDINATES - make it proportional to new image size
    bboxes_new = (bboxes - offsets_prop) / (
        repeat_array(crop_size, n=num_bboxes * 2) / float(img_size))
    
    # FIX NULL DIGIT BBOXES
    num_bboxes_to_fix = 5 - N
    if num_bboxes_to_fix > 0:
        replacement = repeat_array(NULL_BBOX, n=num_bboxes_to_fix, axis=0)
        bboxes_new[4:4 + 4 * num_bboxes_to_fix] = replacement
    
    return im2, bboxes_new


# ==============================================================================
#                                                             BATCH_AUGMENTATION
# ==============================================================================
def batch_augmentation(X,
                       bboxes,
                       N,
                       crop=None,
                       random_crop=False,
                       brightness=None,
                       contrast=None,
                       blur=None,
                       noise=None,
                       seed=None):
    """ Takes an array of images and returns a new array where each image
        has undergone transformations such as the following, based on what
        transformations were chosen:
          - Random crops
          - random rotation
          - Random brightness, contrast, blur, and noise

    Args:
        X:          (array) batch of images
        bboxes:     (array) Array of bboxes data for each image in batch.
        N:          (array) The number of digits in each image in batch.
        crop:       (int) Take a square crop of this dimension.
                    eg: 54, will take 54x54 crops of the images
        random_crop:(bool) should the cropping position be random?
                    If False, then the crops are centered crops.
        brightness: (float) maximum amount of Standard Deviation to be used for
                    brightness centered at 1.
        contrast:   (float) maximum amount of Standard Deviation to be used for
                    contrast centered at 1.
        blur:       (int) max amount of gaussian blur radius.
        noise:      ()

    Returns:
        Tuple of the following two things:
        - X_new
        - bboxes_new
    """
    np.random.seed(seed=seed)
    
    # INITALIZE OUTPUT ARRAYS
    if crop is not None:
        X_new = np.empty(shape=[X.shape[0], crop, crop], dtype=np.uint8)
    else:
        X_new = np.empty(shape=X.shape, dtype=np.uint8)
    bboxes_new = copy.deepcopy(bboxes)
    
    # PERFORM TRANSFORMS ON EACH IMAGE
    for i in range(X.shape[0]):
        im = array2pil(X[i], mode="grey")
        
        if crop is not None:
            im, bboxes_new[i] = labelled_crop(im,
                                              bboxes=bboxes_new[i],
                                              N=N[i],
                                              min_crop=crop,
                                              outsize=[crop, crop],
                                              random=random_crop)
        if brightness is not None:
            im = random_brightness(im, sd=brightness, min=0.7, max=1.5)
        if contrast is not None:
            im = random_contrast(im, sd=contrast, min=0.7, max=2.0)
        if blur is not None:
            im = random_blur(im, min=0, max=blur)
        if noise is not None:
            im = random_noise(im, sd=noise)
        X_new[i] = pil2array(im)
    
    return X_new, bboxes_new
