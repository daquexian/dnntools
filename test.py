#! /usr/bin/env python3

import numpy as np
import json
import os 
import sys
import tempfile
os.environ['GLOG_minloglevel'] = '2' # suprress Caffe verbose prints
import caffe
os.environ['GLOG_minloglevel'] = '1'
import dnntools.caffe_converter as cc


def mkdirs(path):
    if not os.path.exists(path):
        os.makedirs(path)


def download(url, filename):
    import requests
    mkdirs(os.path.dirname(filename))
    with open(filename, "wb") as f:
        print("Downloading {}...".format(filename))
        response = requests.get(url)
        f.write(response.content)


with open("test_settings.json") as f:
    settings = json.load(f)

os.system("adb push {} /data/local/tmp/".format(sys.argv[1]))

for s in settings:
    if not os.path.exists(s['prototxt']):
        download(s['prototxt_url'], s['prototxt'])
    if not os.path.exists(s['caffemodel']):
        download(s['caffemodel_url'], s['caffemodel'])
    net = caffe.Net(s['prototxt'], s['caffemodel'], caffe.TEST)

    input_blob = net.blobs[s['input']]
    output_blob = net.blobs[s['output']]
    input_shape = input_blob.data.shape
    nhwc_shape = (input_shape[0], input_shape[2], input_shape[3], input_shape[1])
    input_blob.data[...] = np.moveaxis(np.reshape(np.arange(input_blob.data.size), nhwc_shape), -1, 1)    # NHWC to NCHW back
    net.forward()
    expected = output_blob.data[0].flatten()   # Only use the first sample when batch size > 1

    daq = os.path.join(tempfile._get_default_tempdir(), next(tempfile._get_candidate_names()))
    txt = os.path.join(tempfile._get_default_tempdir(), next(tempfile._get_candidate_names()))
    cc.convert(s['prototxt'], s['caffemodel'], daq)

    os.system("adb push {} /data/local/tmp/".format(daq))
    os.system("adb shell /data/local/tmp/dnn_infer_simple /data/local/tmp/{} prob".format(os.path.basename(daq)))
    os.system("adb pull /data/local/tmp/result {}".format(txt))
    os.system("adb shell rm /data/local/tmp/result")

    actual = np.loadtxt(txt)
    np.testing.assert_array_almost_equal(expected, actual)
    print('Passed')
