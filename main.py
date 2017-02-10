import os
import numpy as np
import tensorflow as tf
import copy
from six.moves import range

NULL_BBOX = np.array([0.4, 1.3, 0.6, 1.5], dtype=np.float32)
PRINT_WIDTH = 75    # For keeping printouts consistent width

