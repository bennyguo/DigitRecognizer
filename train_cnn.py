import pandas as pd
import numpy as np
import tensorflow as tf

from dataset import DataSet
from logger import Logger


def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.01)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.01, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


MODEL_NAME = 'cnn'
VALIDATE_RATIO = 0.1
LEARNING_RATE = 1e-4
BATCH_SIZE = 100
TRAINING_ITER = 10000
KEEP_PROB = 0.5
dataset = DataSet(VALIDATE_RATIO)
TRAIN_DATA_SIZE = dataset.train_data_size
logger = Logger(MODEL_NAME)

MODEL_INFO = 'validation ratio: %f\ntrain data size: %d\nbatch size: %d\nlearning rate: %f\ntotal iterations: %d\nkeep prob: %f\n' % (VALIDATE_RATIO, TRAIN_DATA_SIZE, BATCH_SIZE, LEARNING_RATE, TRAINING_ITER, KEEP_PROB)
logger.log('==========================================\n' + MODEL_INFO)

x = tf.placeholder(tf.float32, shape=[None, 784], name='x')
y_ = tf.placeholder(tf.float32, shape=[None, 10], name='y_')

x_image = tf.reshape(x, [-1,28,28,1])

W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cross_entropy)

saver = tf.train.Saver()
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for i in range(TRAINING_ITER):
  labels, features = dataset.nextBatch(BATCH_SIZE)
  _, loss = sess.run([train_step, cross_entropy], feed_dict={x: features, y_: labels, keep_prob: KEEP_PROB})
  if i % 100 == 0:
      #ac_train = sess.run(accuracy, feed_dict={x: dataset.train_features, y_: dataset.train_labels, keep_prob: 1.0})
      ac_validate = sess.run(accuracy, feed_dict={x: dataset.validate_features, y_: dataset.validate_labels, keep_prob: 1.0})
      #print 'iteration %d: training data accuracy %f, validation data accuracy %f' % (i, ac_train, ac_validate)
      #logger.log('iteration %d: training data accuracy %f, validation data accuracy %f\n' % (i, ac_train, ac_validate))
      print 'iteration %d: validation data accuracy %f' % (i, ac_validate)
      logger.log('iteration %d: validation data accuracy %f\n' % (i, ac_validate))

ac = sess.run(accuracy, feed_dict={x: dataset.validate_features, y_: dataset.validate_labels, keep_prob: 1.0})
print 'Validation data accuracy: %f' % ac
logger.log('done. validation data accuracy %f\n' % ac)
saved_file = saver.save(sess, 'model/' + MODEL_NAME)