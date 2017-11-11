import sys
from google.protobuf import text_format
from caffe.proto import caffe_pb2
import caffe
import numpy as np
import struct

params = caffe_pb2.NetParameter()
with open('lenet.prototxt') as f:
    text_format.Merge(f.read(), params)

net = caffe.Net('lenet.prototxt', './lenet_iter_10000.caffemodel', caffe.TEST)

f = open(sys.argv[1], 'wb')

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

NONE = 0
RELU = 1

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

ELTWISE_PROD = 0
ELTWISE_SUM = 1
ELTWISE_MAX = 2

supported_type = ['Convolution', 'InnerProduct', 'Pooling', 'Input', 'Softmax']
supported_activation = ['ReLU']

skipped_layers = []

def add_max_pool(f, bottom, pad_left, pad_right, pad_top, pad_bottom, stride_x, stride_y, filter_height, filter_width,
                 activation):
    write_bin_int_seq(f, [MAX_POOL, bottom, PADDING_LEFT, pad_left, PADDING_RIGHT, pad_right, PADDING_TOP, pad_top, PADDING_BOTTOM,
                          pad_bottom, STRIDE_X, stride_x, STRIDE_Y, stride_y, FILTER_HEIGHT, filter_height,
                          FILTER_WIDTH, filter_width, ACTIVATION, activation])

def add_ave_pool(f, bottom, pad_left, pad_right, pad_top, pad_bottom, stride_x, stride_y, filter_height, filter_width,
                 activation):
    write_bin_int_seq(f, [AVE_POOL, bottom, PADDING_LEFT, pad_left, PADDING_RIGHT, pad_right, PADDING_TOP, pad_top,
                          PADDING_BOTTOM, pad_bottom, STRIDE_X, stride_x, STRIDE_Y, stride_y, FILTER_HEIGHT, filter_height,
                          FILTER_WIDTH, filter_width, ACTIVATION, activation])

def add_conv(f, bottom, pad_left, pad_right, pad_top, pad_bottom, stride_x, stride_y, filter_height, filter_width,
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


def add_FC(f, bottom, num_output, activation, weight, bias=None):
    write_bin_int_seq(f, [FC, bottom, NUM_OUTPUT, num_output, ACTIVATION, activation])

    f.write(bin_int(WEIGHT))
    for x in weight.flatten():
        f.write(bin_float(x))

    if bias is not None:
        f.write(bin_int(BIAS))
        for x in bias.flatten():
            f.write(bin_float(x))


def add_softmax(f, bottom, beta):
    write_bin_int_seq(f, [SOFTMAX, bottom, BETA, beta])


def add_add(f, input1, input2_type, input2):
    write_bin_int_seq(f, [ADD, input1, input2_type])
    if input2_type == TENSOR_OP:
        f.write(bin_int(input2))
    elif input2_type == SCALAR_OP:
        f.write(bin_float(input2))


def add_mul(f, input1, input2_type, input2):
    write_bin_int_seq(f, [MUL, input1, input2_type])
    if input2_type == TENSOR_OP:
        f.write(bin_int(input2))
    elif input2_type == SCALAR_OP:
        f.write(bin_float(input2))


def bin_int(n):
    return struct.pack('i', int(n))


def bin_float(n):
    return struct.pack('f', float(n))


def write_bin_int_seq(f, l):
    for x in l:
        f.write(bin_int(x))


def findInplaceActivation(blob_name):
    for layer in params.layer:
        top = layer.top[0].encode('ascii', 'ignore')
        if len(layer.bottom) == 0 or layer.bottom[0].encode('ascii', 'ignore') != blob_name:
            continue
        if layer.type == 'ReLU':
            if top != blob_name:
                continue
            print "RELU", blob_name
            f.write(bin_int(RELU))
            skipped_layers.append(layer.name)
            return RELU
    else:
        return NONE


for i, layer in enumerate(params.layer):
    top = layer.top[0].encode('ascii', 'ignore')

    if layer.type not in supported_type and layer.type not in supported_activation:
        raise ValueError("Not supported layer " + layer.type)

    if layer.type not in supported_type:
        continue

    if i == 0:
        if layer.type != "Input":
            raise ValueError("First layer should be input")

        f.write(bin_int(INPUT))
        param = layer.input_param
        for dim in param.shape[0].dim:
            f.write(bin_int(dim))


    elif layer.type == 'Convolution':
        bottom = layer.bottom[0].encode('ascii', 'ignore')
        param = layer.convolution_param
        pad = param.pad if param.pad != [] else 0
        pad_left = pad_right = pad_top = pad_bottom = pad
        stride = param.stride[0] if param.stride != [] else 1
        stride_x = stride_y = stride
        kernel_size = param.kernel_size[0]
        filter_height = filter_width = kernel_size

        weights = net.params[layer.name][0].data
        swapped_weights = np.swapaxes(np.swapaxes(weights, 1, 3), 1, 2)

        bias = net.params[layer.name][1].data if param.bias_term else None
        activation = findInplaceActivation(top)

        add_conv(f, blobs.index(bottom), pad_left, pad_right, pad_top, pad_bottom, stride_x, stride_y, filter_height, filter_width,
                 param.num_output, activation, swapped_weights, bias)


    elif layer.type == 'Pooling':
        param = layer.pooling_param
        bottom = layer.bottom[0].encode('ascii', 'ignore')

        pad = param.pad
        pad_left = pad_right = pad_top = pad_bottom = pad
        stride = param.stride
        stride_x = stride_y = stride
        kernel_size = param.kernel_size
        filter_height = filter_width = kernel_size
        activation = findInplaceActivation(top)

        if param.pool == CAFFE_POOL_MAX:
            add_max_pool(f, blobs.index(bottom), pad_left, pad_right, pad_top, pad_bottom, stride_x, stride_y,
                         filter_height, filter_width, activation)
        elif param.pool == CAFFE_POOL_AVE:
            add_ave_pool(f, blobs.index(bottom), pad_left, pad_right, pad_top, pad_bottom, stride_x, stride_y,
                         filter_height, filter_width, activation)
        else:
            raise ValueError("Not supported pool type")



    elif layer.type == 'InnerProduct':
        bottom = layer.bottom[0].encode('ascii', 'ignore')
        param = layer.inner_product_param
        input_dim = list(net.blobs[bottom].data.shape)
        weights = net.params[layer.name][0].data
        if len(input_dim) == 4:
            input_dim[0] = param.num_output
            weights = weights.reshape(input_dim)
            weights = np.swapaxes(np.swapaxes(weights, 1, 3), 1, 2)
        bias = None
        if param.bias_term:
            f.write(bin_int(BIAS))
            bias = net.params[layer.name][1].data
        activation = findInplaceActivation(top)

        add_FC(f, blobs.index(bottom), param.num_output, activation, weights, bias)

    elif layer.type == 'Softmax':
        bottom = layer.bottom[0].encode('ascii', 'ignore')
        add_softmax(f, blobs.index(bottom), 1.)

    elif layer.type == 'Dropout':
        bottom = layer.bottom[0].encode('ascii', 'ignore')
        param = layer.dropout_param
        add_mul(f, bottom, SCALAR_OP, 1 - param.dropout_radio)

    elif layer.type == 'Eltwise':
        bottom0 = layer.bottom[0].encode('ascii', 'ignore')
        bottom1 = layer.bottom[1].encode('ascii', 'ignore')
        param = layer.eltwise_param
        if param.operation == ELTWISE_SUM:
            add_add(f, blobs.index(bottom0), TENSOR_OP, blobs.index(bottom1))
        elif param.operation == ELTWISE_PROD:
            add_mul(f, blobs.index(bottom0), TENSOR_OP, blobs.index(bottom1))
        else:
            raise ValueError("Unsupported EltwiseOp " + str(param.operation))


    f.write(bin_int(TOP_NAME))
    for c in top:
        # write as int
        f.write(bin_int(ord(c)))

    f.write(bin_int(STRING_END))

    f.write(bin_int(PARAM_END))
    if top not in blobs:
        blobs.append(top)

f.write(bin_int(LAYER_END))

f.close()
