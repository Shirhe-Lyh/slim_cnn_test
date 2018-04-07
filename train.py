#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 30 19:27:44 2018

@author: shirhe-lyh
"""

"""Train a CNN model to classifying 10 digits.

Example Usage:
---------------
python3 train.py \
    --images_path: Path to the training images (directory).
    --model_output_path: Path to model.ckpt.
"""

import tensorflow as tf

import model

slim = tf.contrib.slim
flags = tf.app.flags

flags.DEFINE_string('record_path', None, 'Path to training tfrecord file.')
flags.DEFINE_string('logdir', None, 'Path to log directory.')
FLAGS = flags.FLAGS


def get_record_dataset(record_path,
                       reader=None, image_shape=[28, 28, 3], 
                       num_samples=50000, num_classes=10):
    """Get a tensorflow record file.
    
    Args:
        
    """
    if not reader:
        reader = tf.TFRecordReader
        
    keys_to_features = {
        'image/encoded': 
            tf.FixedLenFeature((), tf.string, default_value=''),
        'image/format': 
            tf.FixedLenFeature((), tf.string, default_value='jpeg'),
        'image/class/label': 
            tf.FixedLenFeature([1], tf.int64, default_value=tf.zeros([1], 
                               dtype=tf.int64))}
        
    items_to_handlers = {
        'image': slim.tfexample_decoder.Image(shape=image_shape, 
                                              #image_key='image/encoded',
                                              #format_key='image/format',
                                              channels=3),
        'label': slim.tfexample_decoder.Tensor('image/class/label', shape=[])}
    
    decoder = slim.tfexample_decoder.TFExampleDecoder(
        keys_to_features, items_to_handlers)
    
    labels_to_names = None
    items_to_descriptions = {
        'image': 'An image with shape image_shape.',
        'label': 'A single integer between 0 and 9.'}
    return slim.dataset.Dataset(
        data_sources=record_path,
        reader=reader,
        decoder=decoder,
        num_samples=num_samples,
        num_classes=num_classes,
        items_to_descriptions=items_to_descriptions,
        labels_to_names=labels_to_names)


def main(_):
    dataset = get_record_dataset(FLAGS.record_path)
    data_provider = slim.dataset_data_provider.DatasetDataProvider(dataset)
    image, label = data_provider.get(['image', 'label'])
    inputs, labels = tf.train.batch([image, label],
                                    batch_size=64,
                                    allow_smaller_final_batch=True)
    
    cls_model = model.Model(is_training=True, num_classes=10)
    preprocessed_inputs = cls_model.preprocess(inputs)
    prediction_dict = cls_model.predict(preprocessed_inputs)
    loss_dict = cls_model.loss(prediction_dict, labels)
    loss = loss_dict['loss']
    postprocessed_dict = cls_model.postprocess(prediction_dict)
    classes = postprocessed_dict['classes']
    acc = tf.reduce_mean(tf.cast(tf.equal(classes, labels), 'float'))
    tf.summary.scalar('loss', loss)
    tf.summary.scalar('accuracy', acc)
    
    optimizer = tf.train.MomentumOptimizer(learning_rate=0.01, momentum=0.9)
    train_op = slim.learning.create_train_op(loss, optimizer,
                                             summarize_gradients=True)
    
    slim.learning.train(train_op=train_op, logdir=FLAGS.logdir,
                        save_summaries_secs=20, save_interval_secs=120)
    
if __name__ == '__main__':
    tf.app.run()
    