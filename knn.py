import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import time
import getopt
import sys

from dataset import DataSet
from logger import Logger


def train(model_name, n_neighbors, expand):

    N_NEIGHBORS = n_neighbors
    MODEL_NAME = model_name

    dataset = DataSet(expand=expand)
    TRAIN_DATA_SIZE = dataset.train_data_size

    logger = Logger(MODEL_NAME)
    MODEL_INFO = 'train data size: %d, neighbors number: %d\n' % (TRAIN_DATA_SIZE, N_NEIGHBORS)
    logger.log('==========================================\n' + MODEL_INFO)
    print 'Start fitting knn model using ball tree ...'
    t_begin = time.time()
    knn = KNeighborsClassifier(n_neighbors=N_NEIGHBORS, algorithm='ball_tree', leaf_size=100, n_jobs=-1)
    knn.fit(dataset.train_features, dataset.train_labels_origin)
    print 'Fit process time consumed: %f' % (time.time() - t_begin)

    t_begin = time.time()
    validate_accuracy = knn.score(dataset.validate_features[0:500], dataset.validate_labels_origin[0:500])
    print 'Validate accuracy: %f' % validate_accuracy
    print 'Validate process time consumed: %f' % (time.time() - t_begin)

    logger.log('Validate accuracy: %f\n' % validate_accuracy)
    t_begin = time.time()
    pred = knn.predict(dataset.test_features)
    dataset.writeSubmission(pred, '%s_%d_%d' % (MODEL_NAME, N_NEIGHBORS, TRAIN_DATA_SIZE))
    print 'Predict process time consumed: %f' % (time.time() - t_begin)


if __name__ == '__main__':

    try:
        opts, args = getopt.getopt(sys.argv[1:], "n:k:", ['expand'])
    except getopt.GetoptError:
        print 'Command not found.'
        sys.exit()

    model_name = 'knn'
    n_neighbors = 5
    expand = False
    for o, a in opts:
        if o in ['-n']:
            model_name = a
            print 'Using model name: %s.' % a
        elif o in ['-k']:
            if a.isdigit():
                int_a = int(a)
                if int_a > 0:
                    n_neighbors = int_a
                    print 'Using k(neighbors): %d.' % int_a
        elif o in ['--expand']:
            expand = True
            print 'Using expanded data.'
    train(model_name, n_neighbors=n_neighbors, expand=expand)


