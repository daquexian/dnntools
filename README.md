# dnntools

Convert caffemodel to [DNNLibrary](https://github.com/daquexian/DNNLibrary)'s format(I named this format "daq" for my ID "daquexian").

## Usage



Install this package by pip (It only support Python3.5+ for I used type annotation, **and you need install pycaffe first.**) :

```bash
pip3 install dnntools
```

Use it in python like this:
```bash
import dnntools.caffe_converter

dnntools.caffe_converter.convert('models/squeezenet/deploy.prototxt',
                                 'models/squeezenet/squeezenet_v1.1.caffemodel',
                                 'models/squeezenet/squeezenet.daq')
```

Not all layers and properties are supported. But I will work on it. What's more I'm working on [onnx](https://github.com/onnx/onnx/) support. 

Any PRs are welcome :)

## Have a problem?

Some users met segment fault when converting models. One of the possible reasons is incompatible protobuf version. Please refer to [this issue](https://github.com/daquexian/dnntools/issues/5).