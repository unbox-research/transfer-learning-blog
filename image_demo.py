# ______________________________________________________________________________
# Imports

import argparse
import keras
import os

import matplotlib.pyplot as plt
import numpy             as np
import tensorflow        as tf

from keras.models       import Model
from keras.applications import InceptionV3
from keras.models       import Sequential
from keras.layers       import Dense, Dropout, Activation
from keras.optimizers   import SGD

INCEPTIONV3 = InceptionV3()

# ______________________________________________________________________________
# One image demo.

def read_file(file_name):
	"""
	Convert string of .jpg file path to normalized np array for image processing.
	"""
	file_reader   = tf.read_file(file_name, "file_reader")
	image_reader  = tf.image.decode_jpeg(file_reader, channels = 3, name='jpeg_reader')
	float_caster  = tf.cast(image_reader, tf.float32)
	dims_expander = tf.expand_dims(float_caster, 0)
	
	INPUT_HEIGHT  = 224
	INPUT_WIDTH   = 224
	INPUT_MEAN    = 128
	INPUT_STD     = 128
	
	resized       = tf.image.resize_bilinear(dims_expander, [INPUT_HEIGHT, INPUT_WIDTH])
	normalized    = tf.divide(tf.subtract(resized, [INPUT_MEAN]), [INPUT_STD])
	
	sess   = tf.Session()
	result = sess.run(normalized)
	
	return result

guacomole = read_file('guac.jpg')
plt.imshow(guacomole.reshape(224,224,3))
plt.show()

preds = InceptionV3().predict(guacomole)
print(preds)
print(preds.max())
print(preds.argmax(-1))  # Indeed, it got this right!
# https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a  # Image net class labels.
