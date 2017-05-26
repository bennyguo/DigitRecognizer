import numpy as np
import pandas as pd
from os.path import isfile
from expand import expand
from sys import exit


def prepare(origin_data_file, train_data_file, validate_data_file, expanded_train_data_file, validate_proportion):

    if (not isfile(origin_data_file)):
        exit('Origin data file not found, data preparation failed.')

    print 'Preparing data for training ...'
    origin_data = pd.read_csv(origin_data_file)
    validate_data = origin_data.sample(frac=validate_proportion)
    train_data = origin_data.drop(validate_data.index)
    train_data_extra = pd.DataFrame()
    train_data_expand = pd.concat([train_data, train_data_extra])

    train_data.to_csv(train_data_file, index=False)
    validate_data.to_csv(validate_data_file, index=False)
    train_data_expand.to_csv(expanded_train_data_file, index=False)