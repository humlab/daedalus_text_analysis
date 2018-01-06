# -*- coding: utf-8 -*-
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

def get_word_count_per_decade(filename):
    df = read_from_excel(filename) #.reset_index()
    df['decade'] = 10 * (df['year'] / 10).astype(int)
    df = df.loc[~df['year'].isin([1979,1990])].groupby(['decade']).word_count.sum() #.reset_index()
    df.columns = ['word_count']
    return df
