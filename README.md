# dnntools

Convert caffemodel to [DNNLibrary](https://github.com/daquexian/DNNLibrary)'s format(I named this format "daq" for my ID "daquexian").

## Usage

**Please install pycaffe first.**

Install this package by pip (It only support Python3.5+ for I used type annotation) :

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
