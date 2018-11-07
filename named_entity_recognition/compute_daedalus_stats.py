# -*- coding: utf-8 -*-
import ntpath, io
import pandas as pd
from nltk.tokenize import RegexpTokenizer
import common.file_utility as file_utility

def get_year_from_path(path):
    '''  Extracts year part from filename that matches "Daedalus_yyyy_JJ.txt" '''
    head, tail = ntpath.split(path)
    return int((tail or ntpath.basename(head)).split('_')[1])

def count_words(text):
    ''' Count words (tokens) in text using gensim tokenizer. Returns number of tokens. '''
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(text)
    return len(tokens)

def count_words_in_file(filename):
    ''' Count words (tokens) in a text file. Returns number of tokens. '''
    with io.open(filename,'r',encoding='utf8')  as file:
        return count_words(file.read())

def get_stored_classified_locations(filename):
    '''
    Reads all geocoded locations created (and stored) by apply_geocodes.
    Returns subset marked OK and having no undefined value
    '''
    df = file_utility.FileUtility.read_excel(filename, 'Sheet1')
    return df[(~pd.isnull(df).any(axis=1)) & (df.status == 'OK')]

def compute_words_per_year(source_folder):
    '''
    Counts number of words for each file found in source folder.
    Returns a data frame indexed by year, and with a single "word_count" column
    '''
    filenames = file_utility.FileUtility.get_filenames(source_folder)
    data = [(get_year_from_path(path), count_words_in_file(path)) for path in filenames ]

    df = pd.DataFrame({
        'year':  [x[0] for x in data],
        'word_count': [x[1] for x in data]
    })

    df = df.set_index('year')
    # Years 78-79 and 89-90 had a single Daedalus volume spanning both years
    df.loc[1990] = df.loc[1989]
    df.loc[1979] = df.loc[1978]
    df = df.sort_index()

    return df

def get_word_count_per_decade(filename):
    df = file_utility.FileUtility.read_excel(filename, 'Sheet1') #.reset_index()
    df['decade'] = 10 * (df['year'] / 10).astype(int)
    df = df.loc[~df['year'].isin([1979,1990])].groupby(['decade']).word_count.sum() #.reset_index()
    df.columns = ['word_count']
    return df

def compute_tag_statistics(df_location_tags):

    f1 = lambda x: len(x.unique())
    df = df_location_tags.groupby(['year']).agg({
        'reversename': [ 'count', f1 ],
        'country_code': f1
    })
    df.loc[1990] = df.loc[1989]
    df.loc[1979] = df.loc[1978]
    df.columns = ['location_count', 'unique_locations', 'unique_countries']
    df = df.sort_index()
    return df

def compute_statistics(df_word_stat, df_ner_stat):
    df = pd.concat([ df_word_stat, df_ner_stat ], axis=1)
    df['loc/kWords'] = 1000.0 * df['location_count'] / df['word_count']
    df['unique_loc/kWords'] = 1000.0 * df['unique_locations'] / df['word_count']
    df['unique_country/kWords'] = 1000.0 * df['unique_countries'] / df['word_count']
    return df

text_source_folder = "C:\\TEMP\\Daedalus_Clean_Text_1931-2014"
word_count_filename = "./data/Daedalus_Clean_Text_1931-2014_word_count.xlsx"
geocoded_location_tags_filename = './data/Daedalus_1931-2014_geocoded_location_tags.xlsx'
stat_filename = './data/Daedalus_1931-2014_statistics.xlsx'

do_compute_word_stat = True

if do_compute_word_stat:
    df_word_stat = compute_words_per_year(text_source_folder)
    file_utility.FileUtility.save_excel((df_word_stat, 'Sheet1'), word_count_filename)
else:
    df_word_stat = file_utility.FileUtility.read_excel(word_count_filename, 'Sheet1').set_index('year')

df_location_tags = get_stored_classified_locations(geocoded_location_tags_filename)
df_ner_stat = compute_tag_statistics(df_location_tags)
df_stat = compute_statistics(df_word_stat, df_ner_stat)

file_utility.FileUtility.save_excel((df_stat, 'Sheet1'), stat_filename)

