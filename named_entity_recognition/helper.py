# -*- coding: utf-8 -*-
import glob
import os
import pandas as pd

def save_to_excel(df, filename, sheetname='Sheet1'):
    print("Saving to {0}...".format(filename))
    writer = pd.ExcelWriter(filename)
    df.to_excel(writer, sheetname)
    writer.save()

def read_from_excel(filename, sheetname='Sheet1'):
    with open(filename, 'rb') as f:
        df = pd.read_excel(f, sheetname)
    return df

def get_filenames(folder, extension='txt'):
    '''  Returns a list of text files found in given folder '''
    return glob.glob(os.path.join(folder, '*.{}'.format(extension)))

