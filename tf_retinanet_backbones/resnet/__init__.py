from .resnet50 import ResNet50Backbone


def set_defaults(config):
	if 'type' not in config:
		config['type'] = 'resnet50'
	return config

def from_config(config, **kwargs):
	return ResNet50Backbone(set_defaults(config), **kwargs)
