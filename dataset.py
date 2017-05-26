import pandas as pd
import numpy as np
from constant import *
from os.path import isfile
from prepare import prepare

class DataSet():

    train_labels = train_features = None
    validate_labels = validate_features = None
    test_features = None

    train_data_size = None
    validate_data_size = None
    test_data_size = None

    def __init__(self, expand=False):

        files_exist = True
        if (not isfile(TRAIN_DATA_FILE)) or (not isfile(TEST_DATA_FILE)) or (not isfile(VALIDATE_DATA_FILE)):
            files_exist = False
        if expand and (not isfile(EXPANDED_TRAIN_DATA_FILE)):
            files_exist = False
        print 'Data file not detected, starting prepare procedure ...'
        prepare(ORIGIN_DATA_FILE, TRAIN_DATA_FILE, VALIDATE_DATA_FILE, EXPANDED_TRAIN_DATA_FILE, VALIDATE_DATA_PROPORTION)

        train_data = pd.read_csv(EXPANDED_TRAIN_DATA_FILE) if expand else pd.read_csv(TRAIN_DATA_FILE)
        self.train_labels = np.uint8(pd.get_dummies(train_data['label']))
        self.train_features = np.uint8(train_data.drop('label', 1))

        validate_data = pd.read_csv(VALIDATE_DATA_FILE)
        self.validate_labels = np.uint8(pd.get_dummies(validate_data['label']))
        self.validate_features = np.uint8(validate_data.drop('label', 1))

        self.test_features = np.uint8(pd.read_csv(TEST_DATA_FILE))

        self.train_data_size = self.train_labels.shape[0]
        self.validate_data_size = self.validate_labels.shape[0]
        self.test_data_size = self.test_features.shape[0]

        print 'Train data: %d, validate data: %d, test data: %d' % (self.train_data_size, self.validate_data_size, self.test_data_size)

    def trainBatches(self, batch_size):
        train_data_iter = 0
        while True:
            begin = train_data_iter * batch_size
            end = begin + batch_size
            if begin >= self.train_data_size:
                return
            else:
                yield self.train_labels[begin:end], self.train_features[begin:end]
                train_data_iter += 1

    def validateData(self):
        return self.validate_labels, self.validate_features

    def testData(self):
        return self.test_features

    def writeSubmission(self, prediction, filename):
        result = pd.DataFrame()
        result['ImageID'] = range(1, self.test_data_size + 1)
        result['Label'] = prediction
        result.to_csv(SUBMISSION_FILE_BASE + filename + '.csv', index=False)


