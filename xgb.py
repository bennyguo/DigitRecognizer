import xgboost as xgb
from dataset import DataSet
from logger import Logger
import getopt
import time
import sys


def train(model_name, expand):

    MODEL_NAME = model_name
    dataset = DataSet(expand=expand)
    logger = Logger(MODEL_NAME)
    TRAIN_DATA_SIZE = dataset.train_data_size
    MODEL_INFO = 'train data size: %d\n' % (TRAIN_DATA_SIZE)
    logger.log('==========================================\n' + MODEL_INFO)

    begin = time.time()
    xgb_model = xgb.XGBClassifier(learning_rate=0.1, n_estimators=2000, max_depth=6, subsample=0.8, colsample_bylevel=0.3, n_jobs=-1)
    xgb_model.fit(dataset.train_features, dataset.train_labels_origin)
    print 'Fit model time consumed: %f' % (time.time() - begin)

    begin = time.time()
    ac = xgb_model.score(dataset.validate_features, dataset.validate_labels_origin)
    print 'Validate accuracy: %f' % ac
    logger.log('Validate accuracy: %f\n' % ac)
    print 'Validate process time consumed: %f' % (time.time() - begin)

    begin = time.time()
    pred = xgb_model.predict(dataset.test_features)
    dataset.writeSubmission(pred, '%s_%d' % (MODEL_NAME, TRAIN_DATA_SIZE))
    print 'Predict process time consumed: %f' % (time.time() - begin)


if __name__ == '__main__':

    try:
        opts, args = getopt.getopt(sys.argv[1:], "n:", ['expand'])
    except getopt.GetoptError:
        print 'Command not found.'
        sys.exit()

    model_name = 'xgboost'
    expand = False
    for o, a in opts:
        if o in ['-n']:
            model_name = a
            print 'Using model name: %s.' % a
        elif o in ['--expand']:
            expand = True
            print 'Using expanded data.'
    train(model_name, expand=expand)