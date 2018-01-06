# -*- coding: utf-8 -*-
from gensim.models import Word2Vec

import logging
logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s",  level=logging.INFO)

def load_model(filename):
    model = Word2Vec.load(filename)
    return model

def load_model_vector(filename):
    model = load_model(filename)
    word_vectors = model.wv
    del model
    return word_vectors

# FIXME: Consolidate! Copy from Topic Modelling project:
import os
import sys
import time
import pandas as pd
import shutil
import zipfile

__cwd__ = os.path.abspath(__file__) if '__file__' in globals() else os.getcwd()

sys.path.append(__cwd__)

class UtilityRepository:

    def __init__(self, directory):
        self.directory = directory

    def create(self, clear_target_dir=False):

        if os.path.exists(self.directory) and clear_target_dir:
            shutil.rmtree(self.directory)

        if not os.path.exists(self.directory):
            os.makedirs(self.directory)

    def read_excel(self, filename, sheet):
        if not os.path.isfile(filename):
            raise Exception("File {0} does not exist!".format(filename))
        with pd.ExcelFile(filename) as xls:
            return pd.read_excel(xls, sheet)

    def save_excel(self, data, filename):
        with pd.ExcelWriter(filename) as writer:
            for (df, name) in data:
                df.to_excel(writer, name)
            writer.save()

    def data_path(self, filename):
        return os.path.join(self.directory, filename)

    def ts_data_path(self, filename):
        return os.path.join(self.directory, '{}_{}'.format(time.strftime("%Y%m%d%H%M"), filename))

    def data_path_ts(self, path):
        basename, extension = os.path.splitext(path)
        return os.path.join(self.directory, '{}_{}{}'.format(basename, time.strftime("%Y%m%d%H%M"), extension))

    def zip(self, path):
        if not os.path.exists(path):
            print("ERROR: file not found (zip)")
            return
        folder, filename = os.path.split(path)
        basename, _ = os.path.splitext(filename)
        zip_name = os.path.join(folder, basename + '.zip')
        with zipfile.ZipFile(zip_name, mode='w', compression=zipfile.ZIP_DEFLATED) as zf:
            zf.write(path)
        os.remove(path)