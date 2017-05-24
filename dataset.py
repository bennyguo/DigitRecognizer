import pandas as pd
import numpy as np


class DataSet():

    full_data = None
    train_labels = train_features = None
    validate_labels = validate_features = None
    test_features = None

    train_data_size = None
    validate_data_size = None
    test_data_size = None

    train_data_iter = 0

    def __init__(self, validate_ratio = 0.1):
        print 'Initializing data using %f%% as validate set...' % (validate_ratio * 100)
        full_data = pd.read_csv('train.csv')
        full_labels = pd.get_dummies(full_data['label'])
        full_features = full_data.drop('label', 1)

        full_size = full_data.shape[0]
        self.validate_data_size = int(full_size * validate_ratio)
        self.train_data_size = full_size - self.validate_data_size

        self.train_labels = np.uint8(full_labels[0:self.train_data_size])
        self.train_features = np.uint8(full_features[0:self.train_data_size])
        self.validate_labels = np.uint8(full_labels[self.train_data_size:])
        self.validate_features = np.uint8(full_features[self.train_data_size:])

        self.test_features = np.uint8(pd.read_csv('test.csv'))
        self.test_data_size = self.test_features.shape[0]

        print 'Train data: %d, validate data: %d, test data: %d' % (self.train_data_size, self.validate_data_size, self.test_data_size)

    def nextBatch(self, batch_size):
        begin = self.train_data_iter * batch_size
        if begin >= self.train_data_size:
            begin = 0
            self.train_data_iter = 0
        end = begin + batch_size
        self.train_data_iter += 1
        return self.train_labels[begin:end], self.train_features[begin:end]

    def validateData(self):
        return self.validate_labels, self.validate_features

    def testData(self):
        return self.test_features

    def writeSubmission(self, prediction, filename):
        result = pd.DataFrame()
        result['ImageID'] = range(1, self.test_data_size + 1)
        result['Label'] = prediction
        result.to_csv('submissions/' + filename + '.csv')


