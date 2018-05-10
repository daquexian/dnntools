import onnx
import onnx.utils
import onnx.shape_inference
import numpy as np

from onnx import numpy_helper, helper

w = np.arange(1*5*3*3).reshape([1,5,3,3]).astype(np.float32)
W = numpy_helper.from_array(w, 'W')
b = np.arange(5).reshape([5]).astype(np.float32)
B = numpy_helper.from_array(b, 'B')
conv = helper.make_node(
    'Conv',
    ['X', 'W', 'B'],
    ['Y'],
    'conv',
    'conv',
    pads=[1,1,1,1]
)
graph = helper.make_graph(
    [conv],
    'g',
    [
        helper.make_tensor_value_info('X', onnx.TensorProto.FLOAT, [1, 1, 5, 5]),
        helper.make_tensor_value_info('W', onnx.TensorProto.FLOAT, [1, 5, 3, 3]),
        helper.make_tensor_value_info('B', onnx.TensorProto.FLOAT, [5]),
    ],
    [
        helper.make_tensor_value_info('Y', onnx.TensorProto.FLOAT, [1, 1, 5, 5]),
    ],
    [
        W, B
    ]
)

model = helper.make_model(graph)

onnx.checker.check_model(model)

new_model = onnx.shape_inference.infer_shapes(model)
print(new_model)

with open('conv.pb', 'wb') as f:
    f.write(model.SerializeToString())
