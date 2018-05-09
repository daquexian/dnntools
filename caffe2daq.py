#! /usr/bin/python3

import sys
from google.protobuf import text_format
from caffe.proto import caffe_pb2
import caffe
import numpy as np
import model_writer as mw
from model_writer import ModelWriter

CAFFE_POOL_MAX = 0
CAFFE_POOL_AVE = 1
CAFFE_POOL_STOCHASTIC = 2

CAFFE_ELTWISE_PROD = 0
CAFFE_ELTWISE_SUM = 1
CAFFE_ELTWISE_MAX = 2

ACTIVATION_NONE = 0
ACTIVATION_RELU = 1

supported_layers = ['Convolution', 'InnerProduct', 'Pooling', 'Input', 'ReLU', 'Softmax', 'Dropout', 'Eltwise',
                    'BatchNorm', 'Scale', 'Concat', 'Power']
supported_activations = ['ReLU']

skipped_layers = []


def find_inplace_activation(params: caffe_pb2.NetParameter, layer_name: str) -> int:
    for i, layer in enumerate(params.layer):
        if layer.name != layer_name:
            continue
        top_blob_name = layer.top[0]
        for j in range(i + 1, len(params.layer)):
            layerJ = params.layer[j]
            if layerJ.top[0] == top_blob_name:
                if layerJ.type == 'ReLU':
                    # print("RELU", layer_name)
                    skipped_layers.append(layerJ.name)
                    return ACTIVATION_RELU
                break
    return ACTIVATION_NONE


def main():
    params = caffe_pb2.NetParameter()

    with open(sys.argv[1]) as f:
        text_format.Merge(f.read(), params)

    net = caffe.Net(sys.argv[1], sys.argv[2], caffe.TEST)

    out_filename = sys.argv[3] if len(sys.argv) > 3 else 'nnmodel'
    f = open(out_filename, 'wb')

    model_writer = ModelWriter(f)

    for i, layer in enumerate(params.layer):
        if layer.type not in supported_layers:
            raise ValueError("Not supported layer " + layer.type)

        if layer.name in skipped_layers:
            continue

        top_name = layer.top[0]

        if i == 0:
            if layer.type != "Input":
                raise ValueError("First layer should be input")

            param = layer.input_param
            model_writer.add_input(top_name, param.shape[0].dim)

        elif layer.type == 'Convolution':
            bottom_name = layer.bottom[0]
            param = layer.convolution_param
            pad = param.pad[0] if param.pad != [] else 0
            pad_left = pad_right = pad_top = pad_bottom = pad
            stride = param.stride[0] if param.stride != [] else 1
            stride_x = stride_y = stride
            kernel_size = param.kernel_size[0]
            filter_height = filter_width = kernel_size

            weights = net.params[layer.name][0].data
            swapped_weights = np.swapaxes(np.swapaxes(weights, 1, 3), 1, 2)

            bias = net.params[layer.name][1].data if param.bias_term else None  # np.zeros(swapped_weights.shape[0])
            activation = find_inplace_activation(params, top_name)

            model_writer.add_conv(bottom_name, top_name, pad_left, pad_right, pad_top, pad_bottom, stride_x, stride_y,
                                  filter_height, filter_width,
                                  param.num_output, activation, swapped_weights, bias)

        elif layer.type == 'Pooling':
            param = layer.pooling_param
            bottom_name = layer.bottom[0]

            pad = param.pad
            pad_left = pad_right = pad_top = pad_bottom = pad
            stride = param.stride
            stride_x = stride_y = stride
            if param.global_pooling:
                kernel_size = -1
            else:
                kernel_size = param.kernel_size
            filter_height = filter_width = kernel_size
            activation = find_inplace_activation(params, top_name)

            if param.pool == CAFFE_POOL_MAX:
                model_writer.add_max_pool(bottom_name, top_name, pad_left, pad_right, pad_top, pad_bottom, stride_x,
                                          stride_y,
                                          filter_height, filter_width, activation)
            elif param.pool == CAFFE_POOL_AVE:
                model_writer.add_ave_pool(bottom_name, top_name, pad_left, pad_right, pad_top, pad_bottom, stride_x,
                                          stride_y,
                                          filter_height, filter_width, activation)
            else:
                raise ValueError("Not supported pool type")

        elif layer.type == 'InnerProduct':
            bottom_name = layer.bottom[0]
            param = layer.inner_product_param
            input_dim = list(net.blobs[bottom_name].data.shape)
            weights = net.params[layer.name][0].data
            num_output = param.num_output
            if len(input_dim) == 4:
                input_dim[0] = param.num_output
                weights = weights.reshape(input_dim)
                weights = np.swapaxes(np.swapaxes(weights, 1, 3), 1, 2)
            bias = net.params[layer.name][1].data if param.bias_term else None  # np.zeros(num_output)
            activation = find_inplace_activation(params, top_name)

            model_writer.add_FC(bottom_name, top_name, num_output, activation, weights, bias)

        elif layer.type == 'ReLU':
            bottom_name = layer.bottom[0]
            param = layer.relu_param
            model_writer.add_ReLU(bottom_name, top_name, param.negative_slope)

        elif layer.type == 'Softmax':
            bottom_name = layer.bottom[0]
            model_writer.add_softmax(bottom_name, top_name, 1.)

        elif layer.type == 'Dropout':
            pass

        elif layer.type == 'Eltwise':
            bottom0 = layer.bottom[0]
            bottom1 = layer.bottom[1]
            param = layer.eltwise_param
            if param.operation == CAFFE_ELTWISE_SUM:
                model_writer.add_add(bottom0, mw.TENSOR_OP, bottom1, top_name)
            elif param.operation == CAFFE_ELTWISE_PROD:
                model_writer.add_mul(bottom0, mw.TENSOR_OP, bottom1, top_name)
            else:
                raise ValueError("Unsupported EltwiseOp " + str(param.operation))

        elif layer.type == 'BatchNorm':
            bottom_name = layer.bottom[0]
            scale_factor = net.params[layer.name][2].data[0]
            mean = net.params[layer.name][0].data / scale_factor
            var = net.params[layer.name][1].data / scale_factor + 1e-5

            model_writer.add_add(bottom_name, mw.ARRAY_OP, -mean, top_name)
            # Append top into blobs so that the mul will use a new index as input
            # It will be the index of output blob of add
            model_writer.add_mul(top_name, mw.ARRAY_OP, 1 / np.sqrt(var), top_name)

        elif layer.type == 'Scale':
            bottom_name = layer.bottom[0]
            param = layer.scale_param
            multipiler = net.params[layer.name][0].data
            model_writer.add_mul(bottom_name, mw.ARRAY_OP, multipiler, top_name)
            if param.bias_term:
                bias = net.params[layer.name][1].data
                model_writer.add_add(top_name, mw.ARRAY_OP, bias, top_name)

        elif layer.type == 'Concat':
            bottom0 = layer.bottom[0]
            bottom1 = layer.bottom[1]
            param = layer.concat_param
            model_writer.add_concat(bottom0, bottom1, top_name, param.axis)

        elif layer.type == 'Power':
            bottom_name = layer.bottom[0]
            param = layer.power_param
            power, scale, shift = param.power, param.scale, param.shift

            internal_bottom_name = bottom_name
            if scale != 1:
                model_writer.add_mul(internal_bottom_name, mw.SCALAR_OP, scale, top_name)
                internal_bottom_name = top_name
            if shift != 0:
                model_writer.add_add(internal_bottom_name, mw.SCALAR_OP, shift, top_name)
                internal_bottom_name = top_name
            if power != 1:
                raise ValueError('Only power == 1 is supported')

    model_writer.save()


if __name__ == '__main__':
    main()
