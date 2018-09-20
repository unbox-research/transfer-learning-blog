# ______________________________________________________________________________
# Imports
import argparse
import keras
import os

import tensorflow as tf
import numpy      as np

from keras.models       import Model
from keras.applications import InceptionV3
from keras.models       import Sequential
from keras.layers       import Dense, Dropout, Activation
from keras.optimizers   import SGD

INCEPTIONV3 = InceptionV3()

# ______________________________________________________________________________
# One image demo.

if verbose:
	guacomole = read_file('guac.jpg')
	import matplotlib.pyplot as plt
	plt.imshow(result.reshape(224,224,3))
	plt.show()

	preds = INCEPTIONV3.predict(guacomole)
	print(preds.argmax(-1))  # Indeed, it got this right!
	# https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a  # Image net class labels.

# ______________________________________________________________________________
# Model layer inspection.

print(INCEPTIONV3.summary())
print(INCEPTIONV3.get_layer(index=0).name)
print(INCEPTIONV3.get_layer(index=0).input)
print(INCEPTIONV3.get_layer(index=-2).name)
print(INCEPTIONV3.get_layer(index=-2).output)