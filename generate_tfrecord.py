#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 09:02:10 2018

@author: shirhe-lyh
"""

"""Generate tfrecord file from images.

Example Usage:
---------------
python3 train.py \
    --images_path: Path to the training images (directory).
    --output_path: Path to .record.
"""

import glob
import io
import os
import tensorflow as tf

from PIL import Image

flags = tf.app.flags

flags.DEFINE_string('images_path', None, 'Path to images (directory).')
flags.DEFINE_string('output_path', None, 'Path to output tfrecord file.')
FLAGS = flags.FLAGS


def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def int64_list_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def bytes_list_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def float_list_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def create_tf_example(image_path):
    with tf.gfile.GFile(image_path, 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size
    label = int(image_path.split('_')[-1].split('.')[0])
    
    tf_example = tf.train.Example(
        features=tf.train.Features(feature={
            'image/encoded': bytes_feature(encoded_jpg),
            'image/format': bytes_feature('jpg'.encode()),
            'image/class/label': int64_feature(label),
            'image/height': int64_feature(height),
            'image/width': int64_feature(width)}))
    return tf_example


def generate_tfrecord(images_path, output_path):
    writer = tf.python_io.TFRecordWriter(output_path)
    for image_file in glob.glob(images_path):
        tf_example = create_tf_example(image_file)
        writer.write(tf_example.SerializeToString())
    writer.close()
    
    
def main(_):
    images_path = os.path.join(FLAGS.images_path, '*.jpg')
    images_record_path = FLAGS.output_path
    generate_tfrecord(images_path, images_record_path)
    
    
if __name__ == '__main__':
    tf.app.run()