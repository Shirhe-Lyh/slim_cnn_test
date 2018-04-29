#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 27 23:43:30 2018

@author: shirhe-lyh
"""

import tensorflow as tf

from abc import ABCMeta
from abc import abstractmethod

slim = tf.contrib.slim


class BaseModel(object):
    """Abstract base class for any model."""
    __metaclass__ = ABCMeta
    
    def __init__(self, num_classes):
        """Constructor.
        
        Args:
            num_classes: Number of classes.
        """
        self._num_classes = num_classes
        
    @property
    def num_classes(self):
        return self._num_classes
    
    @abstractmethod
    def preprocess(self, inputs):
        """Input preprocessing. To be override by implementations.
        
        Args:
            inputs: A float32 tensor with shape [batch_size, height, width,
                num_channels] representing a batch of images.
            
        Returns:
            preprocessed_inputs: A float32 tensor with shape [batch_size, 
                height, widht, num_channels] representing a batch of images.
        """
        pass
    
    @abstractmethod
    def predict(self, preprocessed_inputs):
        """Predict prediction tensors from inputs tensor.
        
        Outputs of this function can be passed to loss or postprocess functions.
        
        Args:
            preprocessed_inputs: A float32 tensor with shape [batch_size,
                height, width, num_channels] representing a batch of images.
            
        Returns:
            prediction_dict: A dictionary holding prediction tensors to be
                passed to the Loss or Postprocess functions.
        """
        pass
    
    @abstractmethod
    def postprocess(self, prediction_dict, **params):
        """Convert predicted output tensors to final forms.
        
        Args:
            prediction_dict: A dictionary holding prediction tensors.
            **params: Additional keyword arguments for specific implementations
                of specified models.
                
        Returns:
            A dictionary containing the postprocessed results.
        """
        pass
    
    @abstractmethod
    def loss(self, prediction_dict, groundtruth_lists):
        """Compute scalar loss tensors with respect to provided groundtruth.
        
        Args:
            prediction_dict: A dictionary holding prediction tensors.
            groundtruth_lists: A list of tensors holding groundtruth
                information, with one entry for each image in the batch.
                
        Returns:
            A dictionary mapping strings (loss names) to scalar tensors
                representing loss values.
        """
        pass
    
        
class Model(BaseModel):
    """xxx definition."""
    
    def __init__(self,
                 is_training,
                 num_classes):
        """Constructor.
        
        Args:
            is_training: A boolean indicating whether the training version of
                computation graph should be constructed.
            num_classes: Number of classes.
        """
        super(Model, self).__init__(num_classes=num_classes)
        
        self._is_training = is_training
        
    def preprocess(self, inputs):
        """Predict prediction tensors from inputs tensor.
        
        Outputs of this function can be passed to loss or postprocess functions.
        
        Args:
            preprocessed_inputs: A float32 tensor with shape [batch_size,
                height, width, num_channels] representing a batch of images.
            
        Returns:
            prediction_dict: A dictionary holding prediction tensors to be
                passed to the Loss or Postprocess functions.
        """
        preprocessed_inputs = tf.to_float(inputs)
        preprocessed_inputs = tf.subtract(preprocessed_inputs, 128.0)
        preprocessed_inputs = tf.div(preprocessed_inputs, 128.0)
        return preprocessed_inputs
    
    def predict(self, preprocessed_inputs):
        """Predict prediction tensors from inputs tensor.
        
        Outputs of this function can be passed to loss or postprocess functions.
        
        Args:
            preprocessed_inputs: A float32 tensor with shape [batch_size,
                height, width, num_channels] representing a batch of images.
            
        Returns:
            prediction_dict: A dictionary holding prediction tensors to be
                passed to the Loss or Postprocess functions.
        """
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                            activation_fn=tf.nn.relu):
            net = preprocessed_inputs
            net = slim.repeat(net, 2, slim.conv2d, 32, [3, 3], scope='conv1')
            net = slim.max_pool2d(net, [2, 2], scope='pool1')
            net = slim.repeat(net, 2, slim.conv2d, 64, [3, 3], scope='conv2')
            net = slim.max_pool2d(net, [2, 2], scope='pool2')
            net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv3')
            net = slim.flatten(net, scope='flatten')
            net = slim.dropout(net, keep_prob=0.5,
                               is_training=self._is_training)
            net = slim.fully_connected(net, 512, scope='fc1')
            net = slim.fully_connected(net, 512, scope='fc2')
            net = slim.fully_connected(net, self.num_classes, 
                                       activation_fn=None, scope='fc3')
        prediction_dict = {'logits': net}
        return prediction_dict
    
    def postprocess(self, prediction_dict):
        """Convert predicted output tensors to final forms.
        
        Args:
            prediction_dict: A dictionary holding prediction tensors.
            **params: Additional keyword arguments for specific implementations
                of specified models.
                
        Returns:
            A dictionary containing the postprocessed results.
        """
        logits = prediction_dict['logits']
        logits = tf.nn.softmax(logits)
        classes = tf.argmax(logits, axis=1)
        postprecessed_dict = {'classes': classes}
        return postprecessed_dict
    
    def loss(self, prediction_dict, groundtruth_lists):
        """Compute scalar loss tensors with respect to provided groundtruth.
        
        Args:
            prediction_dict: A dictionary holding prediction tensors.
            groundtruth_lists: A list of tensors holding groundtruth
                information with length 2 ([train_labels, test_labels]).
                
        Returns:
            A dictionary mapping strings (loss names) to scalar tensors
                representing loss values.
                
        Raises:
            ValueError: If the length of groundtruth is not 2.
        """
        if len(groundtruth_lists) != 2:
            raise ValueError('The length of groundtruth_lists must be 2.')
            
        batch_size = tf.shape(groundtruth_lists[0])[0]
        logits_dict = self._split_train_test(
            prediction_dict, 'logits', batch_size)
        logits = logits_dict['train_logits']
        slim.losses.sparse_softmax_cross_entropy(
            logits=logits, labels=groundtruth_lists[0])
        loss = slim.losses.get_total_loss()
        loss_dict = {'loss': loss}
        return loss_dict
    
    def _split_train_test(self, split_dict, split_name, split_size):
        """split split_dict to two parts: training part and testing part.
        
        Args:
            split_dict: A dictionary holding tensorflow tensors, may be the 
                prediction_dict or postprossed_dict.
            split_name: The field of split_dict which will be splited.
            split_size: An integer indicating that:
                split_dict[split_name][:split_size] for training and
                split_dict[split_name][split_size:] for testing.
                
        Returns:
            splited_dict: The splited results.
            
        Raises:
            ValueError: If split_dict is not a dictionary or split_name is 
                not a field of split_dict.
        """
        if not isinstance(split_dict, dict):
            raise ValueError('split_dict is not a dictionary.')
        if split_name not in split_dict:
            raise ValueError('split_name is not a field of split_dict.')
            
        split_tensor = split_dict[split_name]
        splited_dict = {'train_' + split_name: split_tensor[:split_size],
                        'test_' + split_name: split_tensor[split_size:]}
        return splited_dict
    
    def accuracy(self, postprocessed_dict, groundtruth_lists):
        """Calculate accuracy.
        
        Args:
            postprocessed_dict: A dictionary containing the postprocessed 
                results
            groundtruth_lists: A list of tensors holding groundtruth
                information, with one for training, and another for testing.
                
        Returns:
            accuracy_dict: A dictionary mapping strings (accuracy names) to
                scalar tensors representing accuracy values.
            
        Raises:
            ValueError: If the length of groundtruth is not 2.
        """
        if len(groundtruth_lists) != 2:
            raise ValueError('The length of groundtruth_lists must be 2.')
            
        train_groundtruth_list = groundtruth_lists[0]
        test_groundtruth_list = groundtruth_lists[1]
        batch_size = tf.shape(train_groundtruth_list)[0]
        classes_dict = self._split_train_test(
            postprocessed_dict, 'classes', batch_size)
        train_classes = classes_dict['train_classes']
        test_classes = classes_dict['test_classes']
        train_accuracy = tf.reduce_mean(
            tf.cast(tf.equal(train_classes, train_groundtruth_list), 'float'))
        test_accuracy = tf.reduce_mean(
            tf.cast(tf.equal(test_classes, test_groundtruth_list), 'float'))
        accuracy_dict  = {'train_accuracy': train_accuracy,
                          'test_accuracy': test_accuracy}
        return accuracy_dict