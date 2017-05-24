import pandas as pd
import numpy as np
import tensorflow as tf

from dataset import DataSet

MODEL_NAME = 'ann'

dataset = DataSet(validate_ratio=0.1)

x = tf.placeholder(tf.float32, shape=[None, 784], name='x')
y_ = tf.placeholder(tf.float32, shape=[None, 10], name='y_')

W = tf.Variable(tf.truncated_normal([784, 10], stddev=0.01), name='W')
b = tf.Variable(tf.zeros([10]), name='b')

#y = tf.nn.softmax(tf.matmul(x,W) + b, name='y')
y = tf.matmul(x,W) + b
#cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.maximum(y, 1e-15)), reduction_indices=[1]))
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy)
#train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

saver = tf.train.Saver()
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for i in range(10001):
  labels, features = dataset.nextBatch(100)
  _, loss = sess.run([train_step, cross_entropy], feed_dict={x: features, y_: labels})
  print 'loss %f' % loss
  if i % 1000 == 0:
      ac = sess.run(accuracy, feed_dict={x: dataset.validate_features, y_: dataset.validate_labels})
      print 'iteration %d, accuracy %f' % (i, ac)

print 'Accuracy: %f' % sess.run(accuracy, feed_dict={x: dataset.validate_features, y_: dataset.validate_labels})

saved_file = saver.save(sess, 'model/' + MODEL_NAME)