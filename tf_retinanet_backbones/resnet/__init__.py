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

from .resnet50 import ResNet50Backbone


def set_defaults(config):
	""" Sets the default values in the backbone configuration dict.
	# Arguments
		config: backbone configuration dict.
	# Returns
		config: backbone configuration dict, merged with default values.
	"""
	if 'type' not in config:
		config['type'] = 'resnet50'
	return config


def from_config(config, **kwargs):
	""" Create a ResNet backbone from a backbone configuration dict.
	# Arguments
		config: backbone configuration dict.
	# Returns
		backbone: ResNet backbone for tf-retinanet.
	"""
	return ResNet50Backbone(set_defaults(config), **kwargs)
