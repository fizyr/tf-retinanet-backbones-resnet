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

import tensorflow as tf

from tf_retinanet.backbones        import Backbone
from tf_retinanet.utils.image      import preprocess_image
from tf_retinanet.models.retinanet import retinanet

from tensorflow.keras.applications import ResNet50


class ResNet50Backbone(Backbone):
	""" Describes backbone information and provides utility functions.
	"""

	def __init__(self, config):
		super(ResNet50Backbone, self).__init__(config)

	def retinanet(self, *args, **kwargs):
		""" Returns a retinanet model using the correct backbone.
		"""
		return resnet50_retinanet(*args, weights=self.weights, modifier=self.modifier, **kwargs)

	def validate(self):
		""" Checks whether the backbone string is correct.
		"""
		allowed_backbones = ['resnet50']
		backbone = self.backbone.split('_')[0]

		if backbone not in allowed_backbones:
			raise ValueError('Backbone (\'{}\') not in allowed backbones ({}).'.format(backbone, allowed_backbones))

	def preprocess_image(self, inputs):
		""" Takes as input an image and prepares it for being passed through the network.
		"""
		# Caffe is the default preprocessing for Resnet in keras_application.
		return preprocess_image(inputs, mode='caffe')

def resnet50_retinanet(submodels, inputs=None, modifier=None, weights='imagenet', **kwargs):
	""" Creates a retinanet model using the ResNet50 backbone.
	Arguments
		submodels: RetinaNetSubmodels.
		inputs:    The inputs to the network (defaults to a Tensor of shape (None, None, 3)).
		modifier:  A function handler which can modify the backbone before using it in retinanet (this can be used to freeze backbone layers for example).
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

	# Freeze the BatchNormalization layers.
	for layer in resnet.layers:
		if str(layer.__class__) == "<class 'tensorflow.python.keras.layers.normalization.BatchNormalization'>": # TODO remove this spaghetti code.
			layer.trainable = False

	# Invoke modifier if given.
	if modifier:
		resnet = modifier(resnet)

	# Get output layers.
	layer_names = ["conv3_block4_out", "conv4_block6_out", "conv5_block3_out"]
	layer_outputs = [resnet.get_layer(name).output for name in layer_names]

	return retinanet(inputs, layer_outputs, submodels, **kwargs)
