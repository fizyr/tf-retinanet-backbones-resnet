"""
Copyright 2017-2019 Fizyr (https://fizyr.com)
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from typing import List, Callable, Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50

from tf_retinanet.backbones        import Backbone
from tf_retinanet.utils.image      import preprocess_image
from tf_retinanet.models.retinanet import retinanet


class ResNet50Backbone(Backbone):
	""" Describes backbone information and provides utility functions.
	"""

	def __init__(self, *args, **kwargs):
		super(ResNet50Backbone, self).__init__(*args, **kwargs)

	def retinanet(self, *args, **kwargs) -> tf.keras.Model:
		""" Returns a retinanet model using the correct backbone.
		"""
		return resnet50_retinanet(*args, **kwargs)

	def preprocess_image(self, inputs: np.ndarray) -> np.ndarray:
		""" Takes as input an image and prepares it for being passed through the network.
		"""
		# Caffe is the default preprocessing for Resnet in keras_application.
		return preprocess_image(inputs, mode='caffe')


def resnet50_retinanet(
	submodels: List[Tuple[str, tf.keras.Model]],
	inputs   : tf.keras.layers.Input                      = None,
	modifier : Callable[[tf.keras.Model], tf.keras.Model] = None,
	weights  : str                                        = 'imagenet',
	**kwargs
) -> tf.keras.Model:
	""" Creates a retinanet model using the ResNet50 backbone.
	Arguments
		submodels: List of RetinaNet submodels.
		inputs:    The inputs to the network (defaults to a Tensor of shape (None, None, 3)).
		modifier:  A callable which can modify the backbone before using it in retinanet (this can be used to freeze backbone layers for example).
		weights:   Weights for the backbone (default is imagenet weights).
	Returns:
		RetinaNet model with ResNet50 backbone.
	"""
	# Choose default input.
	if inputs is None:
		if tf.keras.backend.image_data_format() == 'channels_first':
			inputs = tf.keras.layers.Input(shape=(3, None, None))
		else:
			inputs = tf.keras.layers.Input(shape=(None, None, 3))

	# Create the resnet backbone.
	resnet = ResNet50(
		include_top=False,
		weights=weights,
		input_tensor=inputs,
		input_shape=None,
		pooling=None,
		classes=None,
		**kwargs
	)

	# Disable training on the batch normalization layers.
	for layer in resnet.layers:
		# TODO: replace this with an `isinstance`: https://github.com/tensorflow/tensorflow/issues/37635
		if 'BatchNormalization' in str(layer):
			layer.trainable = False

	# Invoke modifier if given.
	if modifier:
		resnet = modifier(resnet)

	# Get output layers.
	layer_names = ["conv3_block4_out", "conv4_block6_out", "conv5_block3_out"]
	layer_outputs = [resnet.get_layer(name).output for name in layer_names]

	return retinanet(inputs, layer_outputs, submodels, **kwargs)
