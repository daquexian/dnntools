import onnx
from onnx import helper, numpy_helper, shape_inference
import numpy as np
from dnntools import model_writer as mw
from dnntools.model_writer import ModelWriter


def convert(onnx_model: str, dest: str = 'nnmodel.daq') -> None:
    model = onnx.load(onnx_model)
    # for node in model.graph.node:
        # print('-----')
        # print(node)
        # print(node.input)
    # for initializer in model.graph.initializer:
        # print(initializer.name)
        # print(initializer.data_type)
    print(model.graph.input)
    # model = shape_inference.infer_shapes(model)
    # print(model.graph.value_info)


if __name__ == '__main__':
    convert('/home/daquexian/models/squeezenet/model.onnx')