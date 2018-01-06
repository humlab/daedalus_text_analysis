# -*- coding: utf-8 -*-
# %%
import os, ntpath, io
import pandas as pd
import glob
from nltk.tokenize import RegexpTokenizer
from NER_helper import save_to_excel, read_from_excel

# %% Helper functions
def get_filenames(folder):
    '''  Returns a list of text files found in given folder '''
    return glob.glob(os.path.join(folder, '*.txt'))

def get_year_from_path(path):
    '''  Extracts year part from filename that matches "Daedalus_yyyy_JJ.txt" '''
    head, tail = ntpath.split(path)
    return int((tail or ntpath.basename(head)).split('_')[1])

# %% Helper functions for word count

def count_words(text):
    ''' Count words (tokens) in text using gensim tokenizer. Returns number of tokens. '''
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(text)
    return len(tokens)

def count_words_in_file(filename):
    ''' Count words (tokens) in a text file. Returns number of tokens. '''
    with io.open(filename,'r',encoding='utf8')  as file:
        return count_words(file.read())

# %% Classified geocoded locations stored in an Excel file
def get_stored_classified_locations(filename):
    '''
    Reads all geocoded locations created (and stored) by apply_geocodes.
    Returns subset marked OK and having no undefined value
    '''
    df = read_from_excel(filename)
    return df[(~pd.isnull(df).any(axis=1)) & (df.status == 'OK')]

# %% Compute words per year
def compute_words_per_year(source_folder):
    '''
    Counts number of words for each file found in source folder.
    Returns a data frame indexed by year, and with a single "word_count" column
    '''
    filenames = get_filenames(source_folder)
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

def compute_tag_statistics(df_location_tags):

    f1 = lambda x: len(x.unique())
    df = df_location_tags.groupby(['year']).agg({
        'reversename': [ 'count', f1 ],
        'country_code': f1
    })
    # Alternativ:
    #gb = df_location_tags.groupby(['year'])
    #abc = pd.concat([
    #        gb.reversename.agg([ 'count', f1 ]),
    #        gb.country_code.agg(f1)
    #], axis=1)
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

# %%
text_source_folder = "C:\\TEMP\\Daedalus_Clean_Text_1931-2014"
word_count_filename = "./data/Daedalus_Clean_Text_1931-2014_word_count.xlsx"
geocoded_location_tags_filename = './data/Daedalus_1931-2014_geocoded_location_tags.xlsx'
stat_filename = './data/Daedalus_1931-2014_statistics.xlsx'

do_compute_word_stat = True

if do_compute_word_stat:
    df_word_stat = compute_words_per_year(text_source_folder)
    save_to_excel(df_word_stat, word_count_filename)
else:
    df_word_stat = read_from_excel(word_count_filename).set_index('year')

df_location_tags = get_stored_classified_locations(geocoded_location_tags_filename)
df_ner_stat = compute_tag_statistics(df_location_tags)
df_stat = compute_statistics(df_word_stat, df_ner_stat)

save_to_excel(df_stat, stat_filename)

