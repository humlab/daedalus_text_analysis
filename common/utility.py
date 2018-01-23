# -*- coding: utf-8 -*-

import os
import sys
import time
import pandas as pd
import shutil
import zipfile
import glob

__cwd__ = os.path.abspath(__file__) if '__file__' in globals() else os.getcwd()

sys.path.append(__cwd__)

def extend(a, b):
    return a.update(b) or a

def revdict(d):
    return {
        d[k]: k for k in d.keys()
    }

def isfileext(path, extension):
    try:
        _, file_extension = os.path.splitext(path)
        return file_extension == extension
    except:
        return False
class FileUtility:

    def __init__(self, directory):
        self.directory = directory

    def create(self, clear_target_dir=False):

        if os.path.exists(self.directory) and clear_target_dir:
            shutil.rmtree(self.directory)

        if not os.path.exists(self.directory):
            os.makedirs(self.directory)

        return self

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

    #@staticmethod
    #def save_to_excel(df, filename, sheetname='Sheet1'):
    #    print("Saving to {0}...".format(filename))
    #    writer = pd.ExcelWriter(filename)
    #    df.to_excel(writer, sheetname)
    #    writer.save()

    #@staticmethod
    #def read_from_excel(filename, sheetname='Sheet1'):
    #    with open(filename, 'rb') as f:
    #        df = pd.read_excel(f, sheetname)
    #    return df

    def data_path(self, filename):
        return os.path.join(self.directory, filename)

    def ts_data_path(self, filename):
        return os.path.join(self.directory, '{}_{}'.format(time.strftime("%Y%m%d%H%M"), filename))

    def data_path_ts(self, path):
        basename, extension = os.path.splitext(path)
        return os.path.join(self.directory, '{}_{}{}'.format(basename, time.strftime("%Y%m%d%H%M"), extension))

    @staticmethod
    def compress(path):
        if not os.path.exists(path):
            print("ERROR: file not found (zip)")
            return
        folder, filename = os.path.split(path)
        basename, _ = os.path.splitext(filename)
        zip_name = os.path.join(folder, basename + '.zip')
        with zipfile.ZipFile(zip_name, mode='w', compression=zipfile.ZIP_DEFLATED) as zf:
            zf.write(path)
        os.remove(path)

    @staticmethod
    def get_filenames(folder, extension='txt'):
        '''  Returns a list of text files found in given folder '''
        return glob.glob(os.path.join(folder, '*.{}'.format(extension)))