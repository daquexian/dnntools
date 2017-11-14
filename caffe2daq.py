#! /usr/bin/python

import sys
from google.protobuf import text_format
from caffe.proto import caffe_pb2
import caffe
import numpy as np
import struct

params = caffe_pb2.NetParameter()
with open(sys.argv[1]) as f:
    text_format.Merge(f.read(), params)

net = caffe.Net(sys.argv[1], sys.argv[2], caffe.TEST)

out_filename = sys.argv[3] if len(sys.argv) > 3 else 'nnmodel'
f = open(out_filename, 'wb')

blobs = []

CAFFE_POOL_MAX = 0
CAFFE_POOL_AVE = 1
CAFFE_POOL_STOCHASTIC = 2

LAYER_END = 0
CONV = 1
MAX_POOL = 2
AVE_POOL = 3
FC = 4
SOFTMAX = 5
INPUT = 6
MUL = 7
ADD = 8
RELU = 9
CONCAT = 10

ACTIVATION_NONE = 0
ACTIVATION_RELU = 1

PARAM_END = 0
PADDING_LEFT = 1
PADDING_RIGHT = 2
PADDING_TOP = 3
PADDING_BOTTOM = 4
STRIDE_X = 5
STRIDE_Y = 6
FILTER_HEIGHT = 7
FILTER_WIDTH = 8
NUM_OUTPUT = 9
WEIGHT = 10
BIAS = 11
ACTIVATION = 12
TOP_NAME = 13
BETA = 14

STRING_END = 0

TENSOR_OP = 0
SCALAR_OP = 1
ARRAY_OP = 2    # 1d array, for batchnorm

ELTWISE_PROD = 0
ELTWISE_SUM = 1
ELTWISE_MAX = 2

supported_type = ['Convolution', 'InnerProduct', 'Pooling', 'Input', 'ReLU', 'Softmax', 'Dropout', 'Eltwise',
                  'BatchNorm', 'Scale', 'Concat']
supported_activation = ['ReLU']

skipped_layers = []

def blob_index(blob_name):
    # blob.rIndex(blob_name)
    return len(blobs) - blobs[-1::-1].index(blob_name) - 1


def layer_end(blob_name):
    f.write(bin_int(TOP_NAME))
    for c in blob_name:
        # write as int
        f.write(bin_int(ord(c)))

    f.write(bin_int(STRING_END))

    f.write(bin_int(PARAM_END))

    # Append the name of top even when the top is the same as bottom
    # Convert in-place to non in-place in this way
    blobs.append(blob_name)


def add_input(f, top_name, dim):
    f.write(bin_int(INPUT))
    for d in dim:
        f.write(bin_int(d))

    layer_end(top_name)


def add_max_pool(f, bottom, top_name, pad_left, pad_right, pad_top, pad_bottom, stride_x, stride_y, filter_height, filter_width,
                 activation):
    write_bin_int_seq(f, [MAX_POOL, bottom, PADDING_LEFT, pad_left, PADDING_RIGHT, pad_right, PADDING_TOP, pad_top, PADDING_BOTTOM,
                          pad_bottom, STRIDE_X, stride_x, STRIDE_Y, stride_y, FILTER_HEIGHT, filter_height,
                          FILTER_WIDTH, filter_width, ACTIVATION, activation])
    layer_end(top_name)

def add_ave_pool(f, bottom, top_name, pad_left, pad_right, pad_top, pad_bottom, stride_x, stride_y, filter_height, filter_width,
                 activation):
    write_bin_int_seq(f, [AVE_POOL, bottom, PADDING_LEFT, pad_left, PADDING_RIGHT, pad_right, PADDING_TOP, pad_top,
                          PADDING_BOTTOM, pad_bottom, STRIDE_X, stride_x, STRIDE_Y, stride_y, FILTER_HEIGHT, filter_height,
                          FILTER_WIDTH, filter_width, ACTIVATION, activation])
    layer_end(top_name)

def add_conv(f, bottom, top_name, pad_left, pad_right, pad_top, pad_bottom, stride_x, stride_y, filter_height, filter_width,
             num_output, activation, weight, bias=None):
    write_bin_int_seq(f, [CONV, bottom, PADDING_LEFT, pad_left, PADDING_RIGHT, pad_right, PADDING_TOP, pad_top, PADDING_BOTTOM,
                          pad_bottom, STRIDE_X, stride_x, STRIDE_Y, stride_y, FILTER_HEIGHT, filter_height,
                          FILTER_WIDTH, filter_width, NUM_OUTPUT, num_output, ACTIVATION, activation])

    f.write(bin_int(WEIGHT))
    for x in weight.flatten():
        f.write(bin_float(x))

    if bias is not None:
        f.write(bin_int(BIAS))
        for x in bias.flatten():
            f.write(bin_float(x))

    layer_end(top_name)


def add_FC(f, bottom, top_name, num_output, activation, weight, bias=None):
    write_bin_int_seq(f, [FC, bottom, NUM_OUTPUT, num_output, ACTIVATION, activation])

    f.write(bin_int(WEIGHT))
    for x in weight.flatten():
        f.write(bin_float(x))

    if bias is not None:
        f.write(bin_int(BIAS))
        for x in bias.flatten():
            f.write(bin_float(x))

    layer_end(top_name)


def add_ReLU(f, bottom, top_name):
    write_bin_int_seq(f, [RELU, bottom])

    layer_end(top_name)


def add_softmax(f, bottom, top_name, beta):
    write_bin_int_seq(f, [SOFTMAX, bottom, BETA])
    f.write(bin_float(beta))

    layer_end(top_name)


def add_add(f, input1, input2_type, input2, top_name):
    write_bin_int_seq(f, [ADD, input1, input2_type])
    if input2_type == TENSOR_OP:
        f.write(bin_int(input2))
    elif input2_type == SCALAR_OP:
        f.write(bin_float(input2))
    elif input2_type == ARRAY_OP:
        f.write(bin_int(len(input2.flatten())))
        for x in input2.flatten():
            f.write(bin_float(x))

    layer_end(top_name)


def add_mul(f, input1, input2_type, input2, top_name):
    write_bin_int_seq(f, [MUL, input1, input2_type])
    if input2_type == TENSOR_OP:
        f.write(bin_int(input2))
    elif input2_type == SCALAR_OP:
        f.write(bin_float(input2))
    elif input2_type == ARRAY_OP:
        f.write(bin_int(len(input2.flatten())))
        for x in input2.flatten():
            f.write(bin_float(x))

    layer_end(top_name)


def add_concat(f, input1, input2, top_name, axis):
    if axis == 1:
        write_bin_int_seq(f, [CONCAT, 2, input1, input2, 3])

        layer_end(top_name)
    else:
        raise ValueError("Unsupported concat layer's axis")


def bin_int(n):
    return struct.pack('i', int(n))


def bin_float(n):
    return struct.pack('f', float(n))


def write_bin_int_seq(f, l):
    for x in l:
        f.write(bin_int(x))


def findInplaceActivation(layer_name):
    for i, layer in enumerate(params.layer):
        if layer.name != layer_name:
            continue
        top_blob_name = layer.top[0].encode('ascii', 'ignore')
        for j in range(i + 1, len(params.layer)):
            layerJ = params.layer[j]
            if layerJ.top[0].encode('ascii', 'ignore') == top_blob_name:
                if layerJ.type == 'ReLU':
                    print "RELU", layer_name
                    skipped_layers.append(layerJ.name)
                    return ACTIVATION_RELU
                break
    return ACTIVATION_NONE


for i, layer in enumerate(params.layer):
    if layer.type not in supported_type:
        raise ValueError("Not supported layer " + layer.type)

    if layer.name in skipped_layers:
        continue

    top_name = layer.top[0].encode('ascii', 'ignore')

    if i == 0:
        if layer.type != "Input":
            raise ValueError("First layer should be input")

        param = layer.input_param
        add_input(f, top_name, param.shape[0].dim)


    elif layer.type == 'Convolution':
        bottom_name = layer.bottom[0].encode('ascii', 'ignore')
        param = layer.convolution_param
        pad = param.pad[0] if param.pad != [] else 0
        pad_left = pad_right = pad_top = pad_bottom = pad
        stride = param.stride[0] if param.stride != [] else 1
        stride_x = stride_y = stride
        kernel_size = param.kernel_size[0]
        filter_height = filter_width = kernel_size

        weights = net.params[layer.name][0].data
        swapped_weights = np.swapaxes(np.swapaxes(weights, 1, 3), 1, 2)

        bias = net.params[layer.name][1].data if param.bias_term else None #np.zeros(swapped_weights.shape[0])
        activation = findInplaceActivation(top_name)

        add_conv(f, blob_index(bottom_name), top_name, pad_left, pad_right, pad_top, pad_bottom, stride_x, stride_y, filter_height, filter_width,
                 param.num_output, activation, swapped_weights, bias)


    elif layer.type == 'Pooling':
        param = layer.pooling_param
        bottom_name = layer.bottom[0].encode('ascii', 'ignore')

        pad = param.pad
        pad_left = pad_right = pad_top = pad_bottom = pad
        stride = param.stride
        stride_x = stride_y = stride
        kernel_size = param.kernel_size
        filter_height = filter_width = kernel_size
        activation = findInplaceActivation(top_name)

        if param.pool == CAFFE_POOL_MAX:
            add_max_pool(f, blob_index(bottom_name), top_name, pad_left, pad_right, pad_top, pad_bottom, stride_x, stride_y,
                         filter_height, filter_width, activation)
        elif param.pool == CAFFE_POOL_AVE:
            add_ave_pool(f, blob_index(bottom_name), top_name, pad_left, pad_right, pad_top, pad_bottom, stride_x, stride_y,
                         filter_height, filter_width, activation)
        else:
            raise ValueError("Not supported pool type")



    elif layer.type == 'InnerProduct':
        bottom_name = layer.bottom[0].encode('ascii', 'ignore')
        param = layer.inner_product_param
        input_dim = list(net.blobs[bottom_name].data.shape)
        weights = net.params[layer.name][0].data
        num_output = param.num_output
        if len(input_dim) == 4:
            input_dim[0] = param.num_output
            weights = weights.reshape(input_dim)
            weights = np.swapaxes(np.swapaxes(weights, 1, 3), 1, 2)
        bias = net.params[layer.name][1].data if param.bias_term else None #np.zeros(num_output)
        activation = findInplaceActivation(top_name)

        add_FC(f, blob_index(bottom_name), top_name, num_output, activation, weights, bias)

    elif layer.type == 'ReLU':
        bottom_name = layer.bottom[0].encode('ascii', 'ignore')
        param = layer.relu_param
        if param.negative_slope != 0:
            raise ValueError("Non-zero ReLU's negative slope is not supported")
        add_ReLU(f, blob_index(bottom_name), top_name)

    elif layer.type == 'Softmax':
        bottom_name = layer.bottom[0].encode('ascii', 'ignore')
        add_softmax(f, blob_index(bottom_name), top_name, 1.)

    elif layer.type == 'Dropout':
        bottom_name = layer.bottom[0].encode('ascii', 'ignore')
        param = layer.dropout_param
        add_mul(f, blob_index(bottom_name), SCALAR_OP, 1 - param.dropout_ratio, top_name)

    elif layer.type == 'Eltwise':
        bottom0 = layer.bottom[0].encode('ascii', 'ignore')
        bottom1 = layer.bottom[1].encode('ascii', 'ignore')
        param = layer.eltwise_param
        if param.operation == ELTWISE_SUM:
            add_add(f, blob_index(bottom0), TENSOR_OP, blob_index(bottom1), top_name)
        elif param.operation == ELTWISE_PROD:
            add_mul(f, blob_index(bottom0), TENSOR_OP, blob_index(bottom1), top_name)
        else:
            raise ValueError("Unsupported EltwiseOp " + str(param.operation))

    elif layer.type == 'BatchNorm':
        bottom_name = layer.bottom[0].encode('ascii', 'ignore')
        scale_factor = net.params[layer.name][2].data[0]
        mean = net.params[layer.name][0].data / scale_factor
        var = net.params[layer.name][1].data / scale_factor + 1e-5

        add_add(f, blob_index(bottom_name), ARRAY_OP, -mean, top_name)
        # Append top into blobs so that the mul will use a new index as input
        # It will be the index of output blob of add
        add_mul(f, blob_index(top_name), ARRAY_OP, 1 / np.sqrt(var), top_name)

    elif layer.type == 'Scale':
        bottom_name = layer.bottom[0].encode('ascii', 'ignore')
        param = layer.scale_param
        multipiler = net.params[layer.name][0].data
        add_mul(f, blob_index(bottom_name), ARRAY_OP, multipiler, top_name)
        if param.bias_term:
            bias = net.params[layer.name][1].data
            add_add(f, blob_index(top_name), ARRAY_OP, bias, top_name)

    elif layer.type == 'Concat':
        bottom0 = layer.bottom[0].encode('ascii', 'ignore')
        bottom1 = layer.bottom[1].encode('ascii', 'ignore')
        param = layer.concat_param
        if param.axis != 1:
            raise ValueError("Unsupported concat layer's axis " + str(param.axis))
        add_concat(f, blob_index(bottom0), blob_index(bottom1), top_name, param.axis)


f.write(bin_int(LAYER_END))

f.close()
