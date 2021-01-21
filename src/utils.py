# coding=utf-8
# Copyleft 2019 Project LXRT

import sys
import csv
import base64
import time

import numpy as np
import h5py

field2label = {
    "img_h":0, "img_w":1, "objects_id":2, "objects_conf":3,
    "attrs_id":4, "attrs_conf":5, "num_boxes":6, "boxes":7, "features":8
}

def get_imgids_h5(hdf5_name, split='trainval'):
    with h5py.File(hdf5_name, 'r') as hf:
        img_ids = list(hf.keys())
    print(f'[INFO] There are {len(img_ids)} images in {split} set')
    return img_ids

def load_obj_h5(hdf5_name, img_id):
    """
    Online fetch data by image id
    Args:
        hdf5_name:
        img_id:

    Returns:

    """
    datum = dict()
    with h5py.File(hdf5_name, "r") as hf:
        h5data = hf[str(img_id)]
        for k in ['img_h', 'img_w', 'num_boxes']:
            datum[k] = int(h5data[field2label[k]])

        boxes = datum['num_boxes']
        decode_config = [
            ('objects_id', (boxes, ), np.int64),
            ('objects_conf', (boxes, ), np.float32),
            ('attrs_id', (boxes, ), np.int64),
            ('attrs_conf', (boxes, ), np.float32),
            ('boxes', (boxes, 4), np.float32),
            ('features', (boxes, -1), np.float32),
        ]
        for k, shape, dtype in decode_config:
            key = field2label[k]
            datum[k] = np.frombuffer(base64.b64decode(h5data[key]), dtype=dtype)
            datum[k] = datum[k].reshape(shape)
            datum[k].setflags(write=False)
    return datum

csv.field_size_limit(sys.maxsize)
FIELDNAMES = ["img_id", "img_h", "img_w", "objects_id", "objects_conf",
              "attrs_id", "attrs_conf", "num_boxes", "boxes", "features"]
def load_obj_tsv(fname, topk=None):
    """Load object features from tsv file.

    :param fname: The path to the tsv file.
    :param topk: Only load features for top K images (lines) in the tsv file.
        Will load all the features if topk is either -1 or None.
    :return: A list of image object features where each feature is a dict.
        See FILENAMES above for the keys in the feature dict.
    """
    data = []
    start_time = time.time()
    print("Start to load Faster-RCNN detected objects from %s" % fname)
    with open(fname) as f:
        reader = csv.DictReader(f, FIELDNAMES, delimiter="\t")
        for i, item in enumerate(reader):

            for key in ['img_h', 'img_w', 'num_boxes']:
                item[key] = int(item[key])
            
            boxes = item['num_boxes']
            decode_config = [
                ('objects_id', (boxes, ), np.int64),
                ('objects_conf', (boxes, ), np.float32),
                ('attrs_id', (boxes, ), np.int64),
                ('attrs_conf', (boxes, ), np.float32),
                ('boxes', (boxes, 4), np.float32),
                ('features', (boxes, -1), np.float32),
            ]
            for key, shape, dtype in decode_config:
                item[key] = np.frombuffer(base64.b64decode(item[key]), dtype=dtype)
                item[key] = item[key].reshape(shape)
                item[key].setflags(write=False)

            data.append(item)
            if topk is not None and len(data) == topk:
                break
    elapsed_time = time.time() - start_time
    print("Loaded %d images in file %s in %d seconds." % (len(data), fname, elapsed_time))
    return data

