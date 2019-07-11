import tensorflow as tf

from tf_retinanet.backbones        import Backbone
from tf_retinanet.utils.image      import preprocess_image
from tf_retinanet.models.retinanet import retinanet


# TODO wait for tf updates (already there).
# When tensorflow is updated the following lien should be uncommented.
# from tensorflow.keras.applications import ResNet50
# And the lines of the block below should be deleted.
#########################################################################

from keras_applications import resnet

from tensorflow.python.keras.applications import keras_modules_injection
from tensorflow.python.util.tf_export import keras_export


@keras_export(
	'keras.applications.resnet50.ResNet50',
	'keras.applications.resnet.ResNet50',
	'keras.applications.ResNet50'
)
@keras_modules_injection
def ResNet50(*args, **kwargs):
	return resnet.ResNet50(*args, **kwargs)
#########################################################################


class ResNet50Backbone(Backbone):
	""" Describes backbone information and provides utility functions.
	"""
	def __init__(self, backbone):
		super(ResNet50Backbone, self).__init__(backbone)

	def retinanet(self, *args, **kwargs):
		""" Returns a retinanet model using the correct backbone.
		"""
		return resnet50_retinanet(*args, **kwargs)

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

def resnet50_retinanet(num_classes, inputs=None, modifier=None, **kwargs):
	# Choose default input.
	if inputs is None:
		if tf.keras.backend.image_data_format() == 'channels_first':
			inputs = tf.keras.layers.Input(shape=(3, None, None))
		else:
			inputs = tf.keras.layers.Input(shape=(None, None, 3))

	# Create the resnet backbone.
	resnet = ResNet50(
		include_top=False,
		weights='imagenet',
		input_tensor=inputs,
		input_shape=None,
		pooling=None,
		classes=None,
		**kwargs
	)

	# Invoke modifier if given.
	if modifier:
		resnet = modifier(resnet)

	# Get output layers.
	layer_names = ["conv3_block4_out", "conv4_block6_out", "conv5_block3_out"]
	layer_outputs = [resnet.get_layer(name).output for name in layer_names]

	return retinanet(inputs, layer_outputs, num_classes)
