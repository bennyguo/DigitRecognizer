import pandas as pd
import numpy as np
import tensorflow as tf
import getopt
import sys
from os.path import exists
from os import makedirs

from dataset import DataSet
from logger import Logger
from constant import MODEL_FILE_BASE, ADAM_OPTIMIZER_LEARNING_RATE

from cnn_tensors import *


def batchTest(test_data, sess):
    iter = 0
    batch_size = 1000
    test_data_size = test_data.shape[0]
    result = np.array([])
    while True:
        begin = iter * batch_size
        end = begin + batch_size
        if begin >= test_data_size:
            break
        else:
            batch_features = test_data[begin:end]
            pred = sess.run(prediction, feed_dict={x: batch_features, keep_prob: 1.0})
            result = np.append(result, pred)
            iter += 1
    return np.int8(result)


def train(model_name, batch_size, epoch, expand):

    MODEL_NAME = model_name
    BATCH_SIZE = batch_size
    TRAINING_EPOCH = epoch
    KEEP_PROB = 0.5
    dataset = DataSet(expand=expand)

    TRAIN_DATA_SIZE = dataset.train_data_size
    logger = Logger(MODEL_NAME)

    MODEL_INFO = 'train data size: %d\nbatch size: %d\ntotal epoches: %d\nkeep prob: %f\n' % (TRAIN_DATA_SIZE, BATCH_SIZE, TRAINING_EPOCH, KEEP_PROB)
    logger.log('==========================================\n' + MODEL_INFO)

    saver = tf.train.Saver()
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    global_iter = 0
    for i in range(TRAINING_EPOCH):
        for labels, features in dataset.trainBatches(BATCH_SIZE):
            _, loss = sess.run([train_step, cross_entropy], feed_dict={x: features, y_: labels, keep_prob: KEEP_PROB})
            if global_iter % 100 == 0:
                print 'iteration %d: loss %f' % (global_iter, loss)
            global_iter += 1

        ac_validate = sess.run(accuracy, feed_dict={x: dataset.validate_features, y_: dataset.validate_labels, keep_prob: 1.0})
        print 'Epoch %d: validation data accuracy %f' % (i, ac_validate)
        logger.log('Epoch %d: validation data accuracy %f\n' % (i, ac_validate))

    ac = sess.run(accuracy, feed_dict={x: dataset.validate_features, y_: dataset.validate_labels, keep_prob: 1.0})
    print 'Final validation data accuracy: %f' % ac
    logger.log('Done. Validation data accuracy %f\n' % ac)

    pred = batchTest(dataset.test_features, sess)
    dataset.writeSubmission(pred, '%s_%d_%d' % (MODEL_NAME, TRAIN_DATA_SIZE, TRAINING_EPOCH))

    if not exists(MODEL_FILE_BASE):
        makedirs(MODEL_FILE_BASE)
    saved_file = saver.save(sess, MODEL_FILE_BASE + MODEL_NAME)


if __name__ == '__main__':

    try:
        opts, args = getopt.getopt(sys.argv[1:], "n:e:", ['expand'])
    except getopt.GetoptError:
        print 'Command not found.'
        sys.exit()
    model_name = 'cnn'
    epoch = 100
    batch_size = 100
    expand = False
    for o, a in opts:
        if o in ['-n']:
            model_name = a
            print 'Using model name: %s.' % a
        elif o in ['-e']:
            if a.isdigit():
                int_a = int(a)
                if int_a > 0:
                    epoch = int_a
                    print 'Using epoch: %d.' % int_a
        elif o in ['--expand']:
            expand = True
            print 'Using expanded data.'
    train(model_name, batch_size=batch_size, epoch=epoch, expand=expand)