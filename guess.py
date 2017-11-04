import os
import sys

import numpy as np
import tensorflow as tf


os.environ['TF_CPP_MIN_LOG_LEVEL']='2'


image_path = "test_images/my_ferrari.jpg"

#Read image using tensorflow
image_data = tf.gfile.FastGFile(image_path, 'rb').read()
