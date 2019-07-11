from .resnet50 import ResNet50Backbone


def from_config(config, **kwargs):
	return ResNet50Backbone(config['type'], **kwargs)
