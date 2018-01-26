# -*- coding: utf-8 -*-
import os
import sys
import time
import pandas as pd
import shutil
import zipfile
import logging

logger = logging.getLogger(__name__)

__cwd__ = os.path.abspath(__file__) if '__file__' in globals() else os.getcwd()

sys.path.append(__cwd__)

class Utility:

    def __init__(self, directory):
        self.directory = directory

    def create(self, clear_target_dir=False):

        if os.path.exists(self.directory) and clear_target_dir:
            shutil.rmtree(self.directory)

        if not os.path.exists(self.directory):
            os.makedirs(self.directory)

    @staticmethod
    def read_excel(filename, sheet):
        if not os.path.isfile(filename):
            raise Exception("File {0} does not exist!".format(filename))
        with pd.ExcelFile(filename) as xls:
            return pd.read_excel(xls, sheet)

    @staticmethod
    def save_excel(data, filename):
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

    @staticmethod
    def zip(path):
        if not os.path.exists(path):
            logger.error("ERROR: file not found (zip)")
            return
        folder, filename = os.path.split(path)
        basename, _ = os.path.splitext(filename)
        zip_name = os.path.join(folder, basename + '.zip')
        with zipfile.ZipFile(zip_name, mode='w', compression=zipfile.ZIP_DEFLATED) as zf:
            zf.write(path)
        os.remove(path)
