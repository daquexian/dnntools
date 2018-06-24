from typing import IO, Any, List, Union, Callable, Tuple
import functools
import struct
import numpy as np

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
LRN = 11
DEPTH_CONV = 12
STRIDED_SLICE = 13

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
LRN_ALPHA = 15
LRN_BETA = 16
LOCAL_SIZE = 17
GROUP = 18

STRING_END = 0

TENSOR_OP = 0
SCALAR_OP = 1
ARRAY_OP = 2  # 1d array, for batchnorm


def bin_int(n: int) -> bytes:
    return struct.pack('i', int(n))


def bin_float(n: Union[float, int]) -> bytes:
    return struct.pack('f', float(n))


def add_layer(top_name_pos: int = 2) -> Callable:
    """
    A decorator function, run model_writer.layer_end(top_name) in the end of model_writer.add_xxxx
    Check out add_input document for an example
    :param top_name_pos:
    :return:
    """
    def decorator(func):
        def __add_layer(*args, **kwargs):
            func(*args, **kwargs)
            args[0].layer_end(args[top_name_pos])

        return __add_layer

    return decorator


class ModelWriter:
    def __init__(self, file: IO[Any]) -> None:
        self._file = file
        self._blobs = []

    def blob_index(self, blob_name: str) -> int:
        # blob.rIndex(blob_name)
        return len(self._blobs) - self._blobs[-1::-1].index(blob_name) - 1

    def layer_end(self, blob_name: str) -> None:
        self._file.write(bin_int(TOP_NAME))
        for c in blob_name:
            # write as int
            self._file.write(bin_int(ord(c)))

        self._file.write(bin_int(STRING_END))

        self._file.write(bin_int(PARAM_END))

        # Append the name of top even when the top is the same as bottom
        # Convert in-place to non in-place in this way
        self._blobs.append(blob_name)

    @add_layer(top_name_pos=1)
    def add_input(self, top_name: str, dim: List[int]) -> None:
        """
        Check out add_layer document.

        add_input with add_layer(top_name_pos=1) is same as

        self._file.write(bin_int(INPUT))
        for d in dim:
            self._file.write(bin_int(d))
        self.layer_end(top_name)
        """
        self._file.write(bin_int(INPUT))
        for d in dim:
            self._file.write(bin_int(d))

    @add_layer()
    def add_max_pool(self, bottom_name: str, top_name: str,
                     pad_left: int, pad_right: int, pad_top: int, pad_bottom: int,
                     stride_x: int, stride_y: int,
                     filter_height: int, filter_width: int, activation: int) -> None:
        bottom = self.blob_index(bottom_name)
        self.write_bin_int_seq(
            [MAX_POOL, bottom, PADDING_LEFT, pad_left, PADDING_RIGHT, pad_right, PADDING_TOP, pad_top,
             PADDING_BOTTOM,
             pad_bottom, STRIDE_X, stride_x, STRIDE_Y, stride_y, FILTER_HEIGHT, filter_height,
             FILTER_WIDTH, filter_width, ACTIVATION, activation])

    @add_layer()
    def add_ave_pool(self, bottom_name: str, top_name: str,
                     pad_left: int, pad_right: int, pad_top: int, pad_bottom: int,
                     stride_x: int, stride_y: int,
                     filter_height: int, filter_width: int, activation: int) -> None:
        bottom = self.blob_index(bottom_name)
        self.write_bin_int_seq(
            [AVE_POOL, bottom, PADDING_LEFT, pad_left, PADDING_RIGHT, pad_right, PADDING_TOP, pad_top,
             PADDING_BOTTOM, pad_bottom, STRIDE_X, stride_x, STRIDE_Y, stride_y, FILTER_HEIGHT,
             filter_height,
             FILTER_WIDTH, filter_width, ACTIVATION, activation])

    @add_layer()
    def add_dep_conv(self, bottom_name: str, top_name: str,
                     pad_left: int, pad_right: int, pad_top: int, pad_bottom: int,
                     stride_x: int, stride_y: int,
                     filter_height: int, filter_width: int, num_output: int, activation: int,
                     weight: np.ndarray, group: int, bias: np.ndarray = None) -> None:
        bottom = self.blob_index(bottom_name)
        self.write_bin_int_seq([DEPTH_CONV, bottom, PADDING_LEFT, pad_left, PADDING_RIGHT, pad_right, PADDING_TOP, pad_top,
                              PADDING_BOTTOM, pad_bottom, STRIDE_X, stride_x, STRIDE_Y, stride_y, FILTER_HEIGHT, filter_height,
                              FILTER_WIDTH, filter_width, NUM_OUTPUT, num_output, ACTIVATION, activation, GROUP, group])

        self._file.write(bin_int(WEIGHT))
        for x in weight.flatten():
            self._file.write(bin_float(x))

        if bias is not None:
            self._file.write(bin_int(BIAS))
            for x in bias.flatten():
                self._file.write(bin_float(x))

    @add_layer()
    def add_conv(self, bottom_name: str, top_name: str,
                 pad_left: int, pad_right: int, pad_top: int, pad_bottom: int,
                 stride_x: int, stride_y: int,
                 filter_height: int, filter_width: int, num_output: int, activation: int,
                 weight: np.ndarray, bias: np.ndarray = None) -> None:
        bottom = self.blob_index(bottom_name)
        self.write_bin_int_seq([CONV, bottom, PADDING_LEFT, pad_left, PADDING_RIGHT, pad_right, PADDING_TOP, pad_top,
                                PADDING_BOTTOM,
                                pad_bottom, STRIDE_X, stride_x, STRIDE_Y, stride_y, FILTER_HEIGHT, filter_height,
                                FILTER_WIDTH, filter_width, NUM_OUTPUT, num_output, ACTIVATION, activation])

        self._file.write(bin_int(WEIGHT))
        for x in weight.flatten():
            self._file.write(bin_float(x))

        if bias is not None:
            self._file.write(bin_int(BIAS))
            for x in bias.flatten():
                self._file.write(bin_float(x))

    # noinspection PyPep8Naming
    @add_layer()
    def add_FC(self, bottom_name: str, top_name: str,
               num_output: int, activation: int,
               weight: np.ndarray, bias: np.ndarray = None) -> None:
        bottom = self.blob_index(bottom_name)
        self.write_bin_int_seq([FC, bottom, NUM_OUTPUT, num_output, ACTIVATION, activation])

        self._file.write(bin_int(WEIGHT))
        for x in weight.flatten():
            self._file.write(bin_float(x))

        if bias is not None:
            self._file.write(bin_int(BIAS))
            for x in bias.flatten():
                self._file.write(bin_float(x))

    # noinspection PyPep8Naming
    @add_layer()
    def add_ReLU(self, bottom_name: str, top_name: str, negative_slope: float) -> None:
        bottom = self.blob_index(bottom_name)
        if negative_slope != 0:
            raise ValueError("Non-zero ReLU's negative slope is not supported")

        self.write_bin_int_seq([RELU, bottom])

    @add_layer()
    def add_softmax(self, bottom_name: str, top_name: str, beta: float) -> None:
        bottom = self.blob_index(bottom_name)
        self.write_bin_int_seq([SOFTMAX, bottom, BETA])
        self._file.write(bin_float(beta))

    @add_layer(top_name_pos=4)
    def add_add(self, input1: str, input2_type: int, input2: Union[str, np.ndarray], top_name: str) -> None:
        input1_index = self.blob_index(input1)
        self.write_bin_int_seq([ADD, input1_index, input2_type])
        if input2_type == TENSOR_OP:
            input2_index = self.blob_index(input2)
            self._file.write(bin_int(input2_index))
        elif input2_type == SCALAR_OP:
            input2_index = self.blob_index(input2)
            self._file.write(bin_float(input2_index))
        elif input2_type == ARRAY_OP:
            self._file.write(bin_int(len(input2.flatten())))
            for x in input2.flatten():
                self._file.write(bin_float(x))

    @add_layer(top_name_pos=4)
    def add_mul(self, input1: str, input2_type: int, input2: Union[str, np.ndarray], top_name: str) -> None:
        input1_index = self.blob_index(input1)
        self.write_bin_int_seq([MUL, input1_index, input2_type])
        if input2_type == TENSOR_OP:
            input2_index = self.blob_index(input2)
            self._file.write(bin_int(input2_index))
        elif input2_type == SCALAR_OP:
            input2_index = self.blob_index(input2)
            self._file.write(bin_float(input2_index))
        elif input2_type == ARRAY_OP:
            self._file.write(bin_int(len(input2.flatten())))
            for x in input2.flatten():
                self._file.write(bin_float(x))

    @add_layer()
    def add_concat(self, inputs: List[str], top_name: str, axis: int=1) -> None:
        if axis == 1:
            input_indexes = list(map(self.blob_index, inputs))
            self.write_bin_int_seq([CONCAT, len(input_indexes), *input_indexes, 3])
        else:
            raise ValueError("Unsupported concat layer's axis " + str(axis))

    @add_layer()
    def add_LRN(self, bottom_name: str, top_name: str, local_size: int, alpha: float, beta: float) -> None:
        #print(local_size, alpha, beta)
        bottom = self.blob_index(bottom_name)
        self.write_bin_int_seq([LRN, bottom, LOCAL_SIZE, local_size])

        self._file.write(bin_int(LRN_ALPHA))
        self._file.write(bin_float(alpha))
        self._file.write(bin_int(LRN_BETA))
        self._file.write(bin_float(beta))

    @add_layer()
    def add_strided_slice(self, bottom_name: str, top_name: str, slice_dim: List[Tuple[int, int, int]],
                          begin_mask: List[bool], end_mask: List[bool], shrink_axis: List[bool]=None) -> None:
        bottom_index = self.blob_index(bottom_name)

        slice_dim = [x if x is not None else (0, 0, 1) for x in slice_dim]
        starts, ends, strides = zip(*slice_dim)
        begin_mask_int = functools.reduce(int.__or__, [1 << i if begin_mask[i] else 0 for i in range(len(starts))], 0)
        end_mask_int = functools.reduce(int.__or__, [1 << i if end_mask[i] else 0 for i in range(len(ends))], 0)
        if shrink_axis is None:
            shrink_axis = [False] * len(starts)
        shrink_axis_int = functools.reduce(int.__or__, [1 << i if shrink_axis[i] else 0
                                                         for i in range(len(shrink_axis))], 0)
        self.write_bin_int_seq([STRIDED_SLICE, bottom_index, *starts, *ends, *strides,
                                begin_mask_int, end_mask_int, shrink_axis_int])

    def write_bin_int_seq(self, l: List[int]) -> None:
        for x in l:
            self._file.write(bin_int(x))

    def save(self) -> None:
        self._file.write(bin_int(LAYER_END))

        self._file.close()
