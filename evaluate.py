#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  2 14:02:05 2018

@author: shirhe-lyh
"""


"""Evaluate the trained CNN model.

Example Usage:

---------------

python3 evaluate.py \

    --frozen_graph_path: Path to model frozen graph.
"""

import numpy as np
import tensorflow as tf

from captcha.image import ImageCaptcha

flags = tf.app.flags
flags.DEFINE_string('frozen_graph_path', None, 'Path to model frozen graph.')
FLAGS = flags.FLAGS


def generate_captcha(text='1'):
    capt = ImageCaptcha(width=28, height=28, font_sizes=[24])
    image = capt.generate_image(text)
    image = np.array(image, dtype=np.uint8)
    return image


def main(_):
    model_graph = tf.Graph()
    with model_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(FLAGS.frozen_graph_path, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    
    with model_graph.as_default():
        with tf.Session(graph=model_graph) as sess:
            inputs = model_graph.get_tensor_by_name('image_tensor:0')
            classes = model_graph.get_tensor_by_name('classes:0')
            for i in range(10):
                label = np.random.randint(0, 10)
                image = generate_captcha(str(label))
                image_np = np.expand_dims(image, axis=0)
                predicted_label = sess.run(classes, 
                                           feed_dict={inputs: image_np})
                print(predicted_label, ' vs ', label)
            
            
if __name__ == '__main__':
    tf.app.run()
