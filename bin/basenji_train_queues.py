#!/usr/bin/env python
# Copyright 2017 Calico LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================
from __future__ import print_function

import logging
from queue import Queue
import sys
from threading import Thread
import time
import glob
from os import path
import os

import numpy as np
import tensorflow as tf

from basenji import params
from basenji import seqnn
from basenji import shared_flags
from basenji import tfrecord_batcher
from basenji.util import set_logger

FLAGS = tf.app.flags.FLAGS

# /data/genome-attention/tfrecords

def main(_):
  np.random.seed(FLAGS.seed)
  run(dir=FLAGS.dir)


def run(dir):
  set_logger(path.join(dir, "experiment.log"))

  # read parameters
  job = params.read_job_params(path.join(dir, "params.txt"))
  tfr_dir = job["data_dir"]
  test_file = None
  train_file = None 
  test_epoch_batches = job["test_epoch_batches"]
  train_epoch_batches = job["train_epoch_batches"]
  train_epochs = job["train_epochs"]

  if tfr_dir:
    # load data
    data_ops, training_init_op, test_init_op = make_data_ops(
      job,
      tfr_dir=tfr_dir
    )
  elif train_file and test_file:
    data_ops, training_init_op, test_init_op = make_data_ops(
      job,
      train_file=train_file,
      test_file=test_file
    )
  else:
    raise Exception(
      'train and/or test paths missing. Aborting.'
    )

  # initialize model
  model = seqnn.SeqNN()
  model.build_from_data_ops(job, data_ops)

  # launch accuracy compute thread
  acc_queue = Queue()
  acc_thread = AccuracyWorker(acc_queue)
  acc_thread.start()

  # checkpoints
  saver = tf.train.Saver()

  with tf.Session() as sess:
    train_writer = tf.summary.FileWriter(dir + '/train',
                                         sess.graph) if dir else None

    coord = tf.train.Coordinator()
    tf.train.start_queue_runners(coord=coord)

    if FLAGS.restart:
      # load variables into session
      saver.restore(sess, FLAGS.restart)
    else:
      # initialize variables
      t0 = time.time()
      print('Initializing...')
      sess.run(tf.local_variables_initializer())
      sess.run(tf.global_variables_initializer())
      print('Initialization time %f' % (time.time() - t0))

    train_loss = None
    best_loss = None
    early_stop_i = 0

    epoch = 0
    while (train_epochs is not None and epoch < train_epochs) or \
          (train_epochs is None and early_stop_i < FLAGS.early_stop):
      t0 = time.time()

      # save previous
      train_loss_last = train_loss

      # train epoch
      print("Training – epoch: {}".format(epoch))
      sess.run(training_init_op)
      train_loss, steps = model.train_epoch_tfr(sess, train_writer, train_epoch_batches)

      # block for previous accuracy compute
      acc_queue.join()

      # test validation
      print("Validation – epoch: {}".format(epoch))
      sess.run(test_init_op)
      valid_acc = model.test_tfr(sess, test_epoch_batches)

      # consider as best
      best_str = ''
      if best_loss is None or valid_acc.loss < best_loss:
        best_loss = valid_acc.loss
        best_str = ', best!'
        early_stop_i = 0
        model_dir = path.join(dir, "model")
        if not os.path.exists(model_dir):
          os.makedirs(model_dir)
        saver.save(sess, path.join(model_dir, "model_best.tf"))
      else:
        early_stop_i += 1

      # measure time
      epoch_time = time.time() - t0
      if epoch_time < 600:
        time_str = '%3ds' % epoch_time
      elif epoch_time < 6000:
        time_str = '%3dm' % (epoch_time / 60)
      else:
        time_str = '%3.1fh' % (epoch_time / 3600)

      # compute and write accuracy update
      #accuracy_update(epoch, steps, train_loss, valid_acc, time_str, best_str)
      acc_queue.put((epoch, steps, train_loss, valid_acc, time_str, best_str, train_writer))

      # update epoch
      epoch += 1

    # finish queue
    acc_queue.join()

    if FLAGS.logdir:
      train_writer.close()


def accuracy_update(epoch, steps, train_loss, valid_acc, time_str, best_str):
  """Compute and write accuracy update."""

  # compute validation accuracy
  valid_r2 = valid_acc.r2().mean()
  valid_corr = valid_acc.pearsonr().mean()

  # print update
  update_line = 'Train loss: %7.5f,  Valid loss: %7.5f,' % (train_loss, valid_acc.loss)
  update_line += '  Valid R2: %7.5f,  Valid R: %7.5f, Time: %s%s' % (valid_r2, valid_corr, time_str, best_str)
  update_line += '\n========================================================================================='
  print(update_line, flush=True)

  del valid_acc


def make_data_ops(job, train_file=None, test_file=None, tfr_dir=None):
  def make_dataset(loc, mode, is_dir=False):
    """
    Creates the tfrecord dataset.

    This function is now expected to take either some filename string OR a
    list of filename strings as the data source for tfrecord_dataset.
    """
    if is_dir:
      pattern = ''
      if mode == tf.estimator.ModeKeys.TRAIN:
        pattern = 'train-*.tfr'
      elif mode == tf.estimator.ModeKeys.EVAL:
        pattern = 'valid-*.tfr'
      elif mode == tf.estimator.ModeKeys.PREDICT:
        pattern = 'test-*.tfr'
      else:
        raise Exception('unrecognized tfrecord mode. Aborting.')
      pattern = path.join(loc, pattern)

      return tfrecord_batcher.tfrecord_dataset(
          pattern,
          job['batch_size'],
          job['seq_length'],
          job['seq_depth'],
          job['target_length'], 
          job['num_targets'],
          mode=mode,
          repeat=False
      )
    else:
      return tfrecord_batcher.tfrecord_dataset(
          loc,
          job['batch_size'],
          job['seq_length'],
          job.get('seq_depth', 4),
          job['target_length'],
          job['num_targets'],
          mode=mode,
          repeat=False
      )

  if tfr_dir:
    training_dataset = make_dataset(tfr_dir, mode=tf.estimator.ModeKeys.TRAIN, is_dir=True)
    test_dataset = make_dataset(tfr_dir, mode=tf.estimator.ModeKeys.EVAL, is_dir=True)
  else:
    training_dataset = make_dataset(train_file, mode=tf.estimator.ModeKeys.TRAIN)
    test_dataset = make_dataset(test_file, mode=tf.estimator.ModeKeys.EVAL)

  iterator = tf.data.Iterator.from_structure(
      training_dataset.output_types, training_dataset.output_shapes)
  data_ops = iterator.get_next()

  training_init_op = iterator.make_initializer(training_dataset)
  test_init_op = iterator.make_initializer(test_dataset)

  return data_ops, training_init_op, test_init_op


class AccuracyWorker(Thread):
  """Compute accuracy statistics and print update line."""
  def __init__(self, acc_queue):
    Thread.__init__(self)
    self.queue = acc_queue
    self.daemon = True

  def run(self):
    while True:
      try:
        # get args
        epoch, steps, train_loss, valid_acc, time_str, best_str, writer = self.queue.get()

        # compute validation accuracy
        valid_r2 = valid_acc.r2().mean()
        valid_corr = valid_acc.pearsonr().mean()

        # add summary
        r2_summary = tf.Summary(value=[tf.Summary.Value(tag="valid_r2",
                                                        simple_value=valid_r2)])
        writer.add_summary(r2_summary, steps)                                        
        r_summary = tf.Summary(value=[tf.Summary.Value(tag="valid_r",
                                                       simple_value=valid_corr)])
        writer.add_summary(r_summary, steps)                                  
        loss_summary = tf.Summary(value=[tf.Summary.Value(tag="valid_loss",
                                                          simple_value=valid_acc.loss)])   
        writer.add_summary(loss_summary, steps)

        # print update
        update_line = 'Epoch: %3d,  Steps: %7d,  Train loss: %7.5f,  Valid loss: %7.5f,' % (epoch+1, steps, train_loss, valid_acc.loss)
        update_line += '  Valid R2: %7.5f,  Valid R: %7.5f, Time: %s%s' % (valid_r2, valid_corr, time_str, best_str)
        logging.info(update_line)

        # delete predictions and targets
        del valid_acc

      except:
        # communicate error
        print('ERROR: epoch accuracy and progress update failed.', flush=True)

      # communicate finished task
      self.queue.task_done()


if __name__ == '__main__':
  tf.app.run(main)
