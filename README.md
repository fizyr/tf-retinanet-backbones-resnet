# TF-Retinanet-Backbones-Resnet

Wrapper around Resnet implementation present in tf.keras, to make it compatible as backbone of
[tf-retinanet](https://github.com/fizyr/tf-retinanet "tf-retinanet").

This backbone is *not* the same used in [keras-retinanet](https://github.com/fizyr/keras-retinanet "keras-retinanet"),
so it *cannot* be used for porting networks from keras to tensorflow.

## Installation.
Install `tf-retinanet`:

```
git clone https://github.com/fizyr/tf-retinanet.git
cd tf-retinanet
python setup.py install --user
cd ..
```

Install `tf-retinanet-backbones-resnet`:

```
git clone https://github.com/fizyr/tf-retinanet-backbones-resnet.git
cd tf-retinanet-backbones-resnet
python setup.py install --user
cd ..
```

## Contributing

1. Check for open issues or open a fresh issue to start a discussion around a feature idea or a bug. There is a `Contributor Friendly` tag for issues that should be ideal for people who are not very familiar with the codebase yet.
2. Fork this repository on GitHub to start making your changes.
3. Write a test which shows that the bug was fixed or that the feature works as expected.
4. Send a pull request, process feedbacks from mantainers and wait until it gets merged and published.
