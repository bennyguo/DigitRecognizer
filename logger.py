from constant import *
from os import path, makedirs


class Logger():

    log_base = LOG_FILE_BASE
    model_name = None
    log_file = None

    def __init__(self, model_name):
        self.model_name = model_name
        self.log_file = self.log_base + self.model_name + '.log'
        if not path.exists(self.log_base):
            makedirs(self.log_base)

    def log(self, text):
        f = open(self.log_file, 'a')
        f.write(text)
        f.close()