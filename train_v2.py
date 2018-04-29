#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 30 04:29:42 2018

@author: shirhe-lyh
"""

"""Train a CNN model to classifying 10 digits.

Example Usage:
---------------
python3 train.py \
    --train_record_path: Path to training tfrecord file.
    --test_record_path: Path to testing tfrecord file.
    --logdir: Path to log directory.
"""

import tensorflow as tf

import model_v2

slim = tf.contrib.slim
flags = tf.app.flags

flags.DEFINE_string('train_record_path', None, 
                    'Path to training tfrecord file.')
flags.DEFINE_string('test_record_path', None, 
                    'Path to testing tfrecord file.')
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
    train_dataset = get_record_dataset(FLAGS.train_record_path)
    train_data_provider = slim.dataset_data_provider.DatasetDataProvider(
        train_dataset)
    train_image, train_label = train_data_provider.get(['image', 'label'])
    test_dataset = get_record_dataset(FLAGS.train_record_path)
    test_data_provider = slim.dataset_data_provider.DatasetDataProvider(
        test_dataset)
    test_image, test_label = test_data_provider.get(['image', 'label'])
    train_inputs, train_labels = tf.train.batch([train_image, train_label],
                                                batch_size=64,
                                                allow_smaller_final_batch=True)
    test_inputs, test_labels = tf.train.batch([test_image, test_label],
                                              batch_size=64,
                                              allow_smaller_final_batch=True)
    inputs = tf.concat([train_inputs, test_inputs], axis=0)
    labels = [train_labels, test_labels]
    
    cls_model = model_v2.Model(is_training=True, num_classes=10)
    preprocessed_inputs = cls_model.preprocess(inputs)
    prediction_dict = cls_model.predict(preprocessed_inputs)
    loss_dict = cls_model.loss(prediction_dict, labels)
    loss = loss_dict['loss']
    postprocessed_dict = cls_model.postprocess(prediction_dict)
    acc_dict = cls_model.accuracy(postprocessed_dict, labels)
    tf.summary.scalar('loss', loss)
    tf.summary.scalar('train_accuracy', acc_dict['train_accuracy'])
    tf.summary.scalar('test_accuracy', acc_dict['test_accuracy'])
    
    optimizer = tf.train.MomentumOptimizer(learning_rate=0.01, momentum=0.9)
    train_op = slim.learning.create_train_op(loss, optimizer,
                                             summarize_gradients=True)
    
    slim.learning.train(train_op=train_op, logdir=FLAGS.logdir,
                        save_summaries_secs=20, save_interval_secs=120,
                        number_of_steps=6000)
    
if __name__ == '__main__':
    tf.app.run()