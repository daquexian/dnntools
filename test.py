#! /usr/bin/env python3

import numpy as np
import json
import os 
os.environ['GLOG_minloglevel'] = '2' # suprress Caffe verbose prints
import caffe
os.environ['GLOG_minloglevel'] = '1'
import argparse
import tempfile
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


def main():
    parser = argparse.ArgumentParser(description='Test DNNLibrary')
    parser.add_argument('-s', dest='settings', type=str, help='test settings json file, default test_settings_27.json', default='test_settings_27.json')
    parser.add_argument('infer_bin', type=str, help='dnn_infer_simple binary file')
    args = parser.parse_args()

    with open(args.settings) as f:
        settings = json.load(f)

    os.system("adb push {} /data/local/tmp/".format(args.infer_bin))

    for s in settings:
        if not os.path.exists(s['prototxt']):
            download(s['prototxt_url'], s['prototxt'])
        if not os.path.exists(s['caffemodel']):
            download(s['caffemodel_url'], s['caffemodel'])
        net = caffe.Net(s['prototxt'], s['caffemodel'], caffe.TEST)

        input_blob = net.blobs[s['input']]
        output_blob = net.blobs[s['output']]
        input_shape = input_blob.data.shape
        nhwc_shape = (input_shape[0], input_shape[2], input_shape[3], input_shape[1])   # reshape to nhwc so the input is the same as that in dnn_infer_simple
        input_blob.data[...] = np.moveaxis(np.reshape(np.arange(input_blob.data.size), nhwc_shape), -1, 1)    # NHWC to NCHW back
        net.forward()
        expected = output_blob.data[0:1]   # Only use the first sample when batch size > 1
        if s.get('transpose', False):
            expected = np.moveaxis(expected, 1, -1).flatten()
        else:
            expected = expected.flatten()

        daq = os.path.join(tempfile._get_default_tempdir(), next(tempfile._get_candidate_names()))
        txt = os.path.join(tempfile._get_default_tempdir(), next(tempfile._get_candidate_names()))
        cc.convert(s['prototxt'], s['caffemodel'], daq)

        os.system("adb push {} /data/local/tmp/".format(daq))
        os.system("adb shell /data/local/tmp/dnn_infer_simple /data/local/tmp/{} prob".format(os.path.basename(daq)))
        os.system("adb pull /data/local/tmp/result {}".format(txt))
        os.system("adb shell rm /data/local/tmp/result")

        actual = np.loadtxt(txt)

        print('====================')
        try:
            np.testing.assert_array_almost_equal(expected, actual)
            print('{} passed'.format(s['name']))
        except AssertionError as e:
            print('{} failed:'.format(s['name']))
            print(str(e))



if __name__ == '__main__':
    main()
