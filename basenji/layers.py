"""Wrapper code for using commonly-used layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

from basenji import ops

def exp_function(length, decay_constant=0.05):
  """
  """
  X = np.zeros((length, length), dtype=np.float32)
  for i in range(length):
      X[i, :] = np.exp(-1*decay_constant*np.abs(i-(np.arange(length))))
  X -= np.eye(length)
  return tf.convert_to_tensor(X) 

def exp_block(seqs_repr, is_training,
              decay_constants, name=''):
  H = seqs_repr
  length = H.get_shape().as_list()[1]
  batch_size = tf.shape(H)[0]
  seqs_repr_next = H
  for decay_constant in decay_constants:
    A = exp_function(length, decay_constant)
    A = tf.expand_dims(A, axis=0)
    C = tf.matmul(tf.tile(A, multiples=[batch_size, 1, 1]), H)
    seqs_repr_next = tf.concat([seqs_repr_next, C], axis=2)

  tf.logging.info('Exp layer with decay constants {}.'.format(decay_constants))

  return seqs_repr_next

def exp_block_variable(seqs_repr, is_training,
              decay_variable, name=''):
  H = seqs_repr
  length = H.get_shape().as_list()[1]
  batch_size = tf.shape(H)[0]
  contexts = [H]
  for i in range(decay_variable):
    with tf.variable_scope('learned_exponential{}'.format(i), reuse=tf.AUTO_REUSE):
      exp_fn = exp_function(length, 1)
      decay_factor = tf.get_variable(f"decay_factor", [1], 
                                     dtype=tf.float32, 
                                     initializer=tf.random_uniform_initializer(0, 1),
                                     constraint=lambda x: tf.clip_by_value(x, 0, np.infty))
      decay_factor = tf.Print(decay_factor, [decay_factor])
      A = tf.pow(exp_fn, decay_factor)
      A = tf.nn.softmax(A, axis=2)
      A = tf.expand_dims(A, axis=0)
      C = tf.matmul(tf.tile(A, multiples=[batch_size, 1, 1]), H)
      contexts.append(C)
  seqs_repr_next = tf.concat(contexts, axis=2)

  tf.logging.info(f'Exp layer with {decay_variable} decay variables.')

  return seqs_repr_next


def multi_head_attention_block(seqs_repr, is_training, num_heads, num_units, n_query_layers,
                               decay_variable, decay_constant, 
                               dropout, query_dropout, 
                               l2_scale, name=''):
  contexts = [seqs_repr]
  for i in range(num_heads):
    with tf.variable_scope('multi_attention{}'.format(i), reuse=tf.AUTO_REUSE):
      context = attention_block(seqs_repr=seqs_repr, 
                                       is_training=is_training, 
                                       n_query_layers=n_query_layers,
                                       decay_variable=decay_variable,
                                       decay_constant=decay_constant,
                                       dropout=dropout,
                                       query_dropout=query_dropout,
                                       l2_scale=l2_scale,
                                       dense=False)
      contexts.append(context)
      tf.logging.info("Adding attention head.")
  seqs_repr = tf.concat(contexts, axis=2)
  tf.logging.info("Concatentating contexts.")

  #with tf.variable_scope('multi_attention_final', reuse=tf.AUTO_REUSE):
    #seqs_repr = tf.layers.dense(inputs=seqs_repr,
    #                              units=num_units,
    #                              activation=tf.nn.relu,
    #                              kernel_initializer=tf.variance_scaling_initializer(scale=2.0, mode='fan_in'),
    #                              kernel_regularizer=tf.contrib.layers.l2_regularizer(l2_scale))
    #seqs_repr = tf.layers.conv1d(
    #              seqs_repr,
    #              filters=2048,
    #              kernel_size=[1],
    #              strides=1,
    #              padding='same',
    #              use_bias=True,
    #              activation=tf.nn.relu,
    #              kernel_initializer=tf.variance_scaling_initializer(scale=2.0, mode='fan_in'),
    #              kernel_regularizer=tf.contrib.layers.l2_regularizer(l2_scale))
    #tf.logging.info("Adding multi-head final dense.")

  return seqs_repr

def dense_attention_block(seqs_repr, is_training, num_layers,
                          decay_variable, decay_constant, 
                          units, dropout, query_dropout, 
                          l2_scale, name=''):
  """
  """
  for i in range(num_layers):
    with tf.variable_scope('dense_attention{}'.format(i), reuse=tf.AUTO_REUSE):
      #seqs_repr = tf.Print(seqs_repr, [tf.shape(seqs_repr)], "{}".format(i))
      seqs_repr = attention_block(seqs_repr, 
                                         is_training, 
                                         decay_variable,
                                         decay_constant,
                                         dropout,
                                         query_dropout,
                                         l2_scale)
      layer_reprs.append(seqs_repr)
  return seqs_repr


def attention_block(seqs_repr, is_training, n_query_layers,
                    decay_variable, decay_constant, 
                    dropout, query_dropout, 
                    l2_scale, name='', dense=True):
  """

  Args:
    seqs_repr: [batchsize, length, num_channels] input sequence
    is_training: whether is a training graph or not
    batch_norm: whether to use batchnorm
    bn_momentum: batch norm momentum
    batch_renorm: whether to use batch renormalization in batchnorm
    l2_scale: L2 weight regularization scale
    name: optional name for the block
  """
  H = seqs_repr 
  length = H.get_shape().as_list()[1]
  num_channels = H.get_shape().as_list()[2]
  Q = H 
  for i in range(n_query_layers):
    Q = tf.layers.dense(Q, num_channels, 
                        activation=tf.nn.tanh,
                        use_bias=True,
                        kernel_initializer= tf.variance_scaling_initializer(scale=2.0, mode='fan_in'), 
                        bias_initializer=tf.zeros_initializer(),
                        kernel_regularizer=None) 

  if query_dropout > 0:
    Q = tf.layers.dropout(inputs=Q,
                          rate=query_dropout,
                          training=is_training)

    tf.logging.info('Query Dropout w/ probability %.3f' % query_dropout)

  A = tf.matmul(Q, H, transpose_b=True)
  
  if decay_variable:
    tf.logging.info("Adding decay variable.")
    exp_fn = exp_function(length, 1)
    decay_factor = tf.get_variable("decay_factor", [1], 
                                   dtype=tf.float32, 
                                   initializer=tf.random_uniform_initializer(0, 1),
                                   constraint=lambda x: tf.clip_by_value(x, 0, np.infty))
    exp_fn = tf.pow(exp_fn, decay_factor)
    A = tf.multiply(A, exp_fn)

  elif decay_constant > 0:
    tf.logging.info("Adding decay constant of {}".format(decay_constant))
    exp_fn = exp_function(length, decay_constant)
    A = tf.multiply(A, exp_fn)
  
  A = tf.nn.softmax(A, axis=2)
  C = tf.matmul(A, H)

  if dense:
    seqs_repr_next = tf.concat([H, C], axis=2)
  else:
    seqs_repr_next = C
  
  if dropout > 0:
    seqs_repr_next = tf.layers.dropout(
                  inputs=seqs_repr_next,
                  rate=dropout,
                  training=is_training)
    tf.logging.info('Dropout w/ probability %.3f' % dropout)

  tf.logging.info('Attention Layer.')
  return seqs_repr_next


def conv_block(seqs_repr, conv_params, is_training,
               batch_norm, batch_norm_momentum,
               batch_renorm, batch_renorm_momentum,
               l2_scale, layer_reprs, name=''):
  """Construct a single (dilated) CNN block.

  Args:
    seqs_repr: [batchsize, length, num_channels] input sequence
    conv_params: convolution parameters
    is_training: whether is a training graph or not
    batch_norm: whether to use batchnorm
    bn_momentum: batch norm momentum
    batch_renorm: whether to use batch renormalization in batchnorm
    l2_scale: L2 weight regularization scale
    name: optional name for the block

  Returns:
    updated representation for the sequence
  """
  # ReLU
  seqs_repr_next = tf.nn.relu(seqs_repr)
  tf.logging.info('ReLU')

  # Convolution
  seqs_repr_next = tf.layers.conv1d(
      seqs_repr_next,
      filters=conv_params.filters,
      kernel_size=[conv_params.filter_size],
      strides=conv_params.stride,
      padding='same',
      dilation_rate=[conv_params.dilation],
      use_bias=False,
      kernel_initializer=tf.variance_scaling_initializer(scale=2.0, mode='fan_in'),
      kernel_regularizer=tf.contrib.layers.l2_regularizer(l2_scale))
  tf.logging.info('Convolution w/ %d %dx%d filters strided %d, dilated %d' %
                  (conv_params.filters, seqs_repr.shape[2],
                   conv_params.filter_size, conv_params.stride,
                   conv_params.dilation))

  # Batch norm
  if batch_norm:
    seqs_repr_next = tf.layers.batch_normalization(
        seqs_repr_next,
        momentum=batch_norm_momentum,
        training=is_training,
        renorm=batch_renorm,
        renorm_clipping={'rmin': 1./4, 'rmax':4., 'dmax':6.},
        renorm_momentum=batch_renorm_momentum,
        fused=True)
    tf.logging.info('Batch normalization')

  # Dropout
  if conv_params.dropout > 0:
    seqs_repr_next = tf.layers.dropout(
        inputs=seqs_repr_next,
        rate=conv_params.dropout,
        training=is_training)
    tf.logging.info('Dropout w/ probability %.3f' % conv_params.dropout)

  # Skip
  if conv_params.skip_layers > 0:
    if conv_params.skip_layers > len(layer_reprs):
      raise ValueError('Skip connection reaches back too far.')

    # Add
    seqs_repr_next += layer_reprs[-conv_params.skip_layers]

  # Dense
  elif conv_params.dense:
    seqs_repr_next = tf.concat(values=[seqs_repr, seqs_repr_next], axis=2)

  # Pool
  if conv_params.pool > 1:
    seqs_repr_next = tf.layers.max_pooling1d(
        inputs=seqs_repr_next,
        pool_size=conv_params.pool,
        strides=conv_params.pool,
        padding='same')
    tf.logging.info('Max pool %d' % conv_params.pool)

  return seqs_repr_next
