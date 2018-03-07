# -*- coding: utf-8 -*-
import os
import gensim
from gensim.models import Word2Vec, LdaModel
from sklearn.feature_extraction import DictVectorizer
import pandas as pd
import numpy as np
from scipy import spatial
import glob
import re
import logging
logger = logging.getLogger(__name__)

current_excel_models = None

class ModelUtility:

    @staticmethod
    def get_model_names(source_folder):
        return [ os.path.split(x)[1] for x in glob.glob(os.path.join(source_folder, '*')) ]
        #return [ os.path.split(x[0])[-1] for x in os.walk(source_folder) if len(x[0]) > 10 ]

    @staticmethod
    def get_excel_models(data_folder):
        global current_excel_models
        if current_excel_models is None:
            current_excel_models = { z: {} for z in ModelUtility.get_model_names(data_folder) }
        return current_excel_models

    @staticmethod
    def export_excel_to_csv(data_folder, basename, excel_model):
        for sheet in excel_model.keys():
            if not hasattr(excel_model[sheet], 'to_csv'):
                continue
            csv_filename = os.path.join(data_folder, '{}/'.format(basename), 'result_{}_{}.csv'.format(basename, sheet))
            if os.path.exists(csv_filename):
                continue
            excel_model[sheet].to_csv(csv_filename, sep='\t')

    @staticmethod
    def load_excel_model(data_folder, basename, sheet):
        filename = os.path.join(data_folder, '{}/'.format(basename), 'result_{}.xlsx'.format(basename))
        with pd.ExcelFile(filename) as xls:
            model = pd.read_excel(xls, sheet)
        return model

    @staticmethod
    def get_excel_model(data_folder, basename):
        if len(ModelUtility.get_excel_models(data_folder).get(basename, {}).keys()) == 0:
            logger.info('Loading {}, please wait...'.format(basename.upper()))
            excel_model = ModelUtility.load_excel_model(data_folder, basename, None)
            excel_model['basename'] = basename
            excel_model['topic_ids'] = excel_model['topic_token_weights'].groupby("topic_id").groups
            ModelUtility.get_excel_models(data_folder)[basename] = excel_model
            logger.info('Exporting CSV for {}, please wait...'.format(basename.upper()))
            ModelUtility.export_excel_to_csv(data_folder, basename, excel_model)
            logger.info('Model {} loaded...'.format(basename.upper()))
        return ModelUtility.get_excel_models(data_folder).get(basename)

    @staticmethod
    def get_result_model_sheet(data_folder, basename, sheet):
        if ModelUtility.get_excel_models(data_folder).get(basename).get(sheet, None) is None:
            logger.info('Model not loaded. Loading...')
            csv_filename = os.path.join(data_folder, '{}/'.format(basename), 'result_{}_{}.csv'.format(basename, sheet))
            if os.path.exists(csv_filename):
                # logger.info('Loading CSV...')
                ModelUtility.get_excel_models(data_folder)\
                    .get(basename)[sheet] = pd.read_csv(csv_filename, sep='\t')
            else:
                # logger.info('Loading EXCEL...')
                _ = ModelUtility.get_excel_model(data_folder, basename)
        else:
            logger.info("Using CACHE data...")
        return ModelUtility.get_excel_models(data_folder).get(basename).get(sheet)

    
    @staticmethod
    def get_topic_keys(data_folder, basename):
        df = ModelUtility.get_result_model_sheet(data_folder, basename, 'topic_tokens')
        df = df.set_index('topic_id')
        return df
    
    @staticmethod
    def get_document_topic_weights_pivot(df_data, aggregate_key, pivot_column='year'):
        df_temp = pd.pivot_table(
            df_data,
            values='weight',
            index=[pivot_column],
            columns=['topic_id'],
            aggfunc=aggregate_key
        ).reset_index() #.set_index('year')
        return df_temp

    @staticmethod
    def get_topic_titles(topic_token_weights, topic_id=None, n_words=100):
        df_temp = topic_token_weights if topic_id is None else topic_token_weights[(topic_token_weights.topic_id==topic_id)]
        df = df_temp\
                .sort_values('weight', ascending=False)\
                .groupby('topic_id')\
                .apply(lambda x: ' '.join(x.token[:n_words].str.title()))
        return df

    @staticmethod
    def get_topic_title(topic_token_weights, topic_id, n_words=100):
        return ModelUtility.get_topic_titles(topic_token_weights, topic_id, n_words=n_words).iloc[0]

    get_topics_tokens_as_text = get_topic_titles
    get_topic_tokens_as_text = get_topic_title

    @staticmethod
    def get_topic_tokens(topic_token_weights, topic_id=None, n_words=100):
        df_temp = topic_token_weights if topic_id is None else topic_token_weights[(topic_token_weights.topic_id == topic_id)]
        df = df_temp.sort_values('weight', ascending=False)[:n_words]
        return df
    
    @staticmethod
    def load_model_vector(filename):
        model = LdaModel.load(filename)
        word_vectors = model.wv
        del model
        return word_vectors

    @staticmethod
    def get_corpus_documents(data_folder, basename):
        # corpus = corpora.MmCorpus(os.path.join(data_folder, basename, 'corpus.mm'))
        # doc_length = pd.DataFrame(dict(length=[ len(x) for x in corpus]))
        filename = os.path.join(data_folder, basename, 'document_index.csv')
        corpus_documents = pd.read_csv(filename, sep='\t', header=0)
        return corpus_documents
       
    @staticmethod
    def compute_topic_terms_vector_space_deprecated(lda, n_words=100):
        '''
        Computes a vector space based on top n_words words for each topics.
        Since the subset of words differs, and their positions differs between topics
        they need to be aligned in common space so that
         - each vector has the same dimension (i.e. number of unique top n tokens over all topics)
         - each token has the same position within that space
        This is with the scikit-learn DictVectorizer (see
            http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.DictVectorizer.html
        )
        '''
        n_topics = lda.num_topics
        ''' Create a term-weight dictionary for each topic '''
        rows = [ { x[0]: x[1] for x in lda.show_topic(i, n_words) } for i in range(0, n_topics) ]
        ''' Use DictVectorizer to align the terms so that each position in resulting vector are the same word '''
        v = DictVectorizer()
        X = v.fit_transform(rows)
        return X, v.get_feature_names()

    @staticmethod
    def compute_and_align_vector_space(feature_data):
        '''
        feature_data is an unaligned list of dicts [ { t1: v, ..., tn: v }, ..., { t1': v1', ..., tn': vn' } ]
        
         DictVectorizer to align the terms so that each position in resulting vector are the same token,
         and create a sparse result matrix
        '''
        v = DictVectorizer()
        X = v.fit_transform(feature_data)
        return X, v.get_feature_names()

    @staticmethod
    def compute_topic_proportions(document_topic_weights, doc_length_series):

        '''
        Topic proportions are computed in the same as in LDAvis.

        Computes topic proportions over entire corpus.
        The document topic weight (pivot) matrice is multiplies by the length of each document
          i.e. each weight are multiplies ny the document's length.
        The topic frequency is then computed by summing up all values over each topic
        This value i then normalized by total sum of matrice

        theta matrix: with each row containing the probability distribution
          over topics for a document, with as many rows as there are documents in the
          corpus, and as many columns as there are topics in the model.

        doc_length integer vector containing token count for each document in the corpus

        '''
        # compute counts of tokens across K topics (length-K vector):
        # (this determines the areas of the default topic circles when no term is highlighted)
        # topic.frequency <- colSums(theta * doc.length)
        # topic.proportion <- topic.frequency/sum(topic.frequency)

        theta = pd.pivot_table(
            document_topic_weights,
            values='weight',
            index=['document_id'],
            columns=['topic_id']
        ) #.set_index('document_id')

        theta_mult_doc_length = theta.mul(doc_length_series.length, axis=0)

        topic_frequency = theta_mult_doc_length.sum()
        topic_proportion = topic_frequency / topic_frequency.sum()

        return topic_proportion
    
    @staticmethod
    def load_dictionary(data_folder, basename):
        filename = os.path.join(data_folder, basename, 'corpus.dict.gz')
        dictionary = gensim.corpora.Dictionary.load(filename)
        return dictionary

    @staticmethod
    def store_dictionary(data_folder, basename, dictionary):
        filename = os.path.join(data_folder, basename, 'corpus.dict.gz')
        dictionary.save(filename)

    @staticmethod
    def store_corpus(data_folder, basename, corpus):
        filename = os.path.join(data_folder, basename, 'corpus.mm')
        gensim.corpora.MmCorpus.serialize(filename, corpus)

    @staticmethod
    def load_corpus(data_folder, basename):
        filename = os.path.join(data_folder, basename, 'corpus.mm')
        corpus = gensim.corpora.MmCorpus(filename)
        return corpus
'''
Helpers (temporary) for adding alpha to token index
'''
def _read_topic_keys(data_folder, basename):
    filename = os.path.join(data_folder, basename, 'topickeys.txt')
    if os.path.isfile(filename):
        df = pd.read_csv(filename, sep='\t', header=None, names=['topic_id', 'alpha', 'tokens'], decimal=b',')
        return df
    return None

def _read_result_model_sheet(csv_filename):
    df = pd.read_csv(csv_filename, sep='\t', header=None, names=['topic_id', 'tokens'])
    df = df[df.topic_id!='alpha']
    df.topic_id = df.topic_id.astype(int)
    return df

def _save_result_model_sheet(csv_filename):
    df = pd.to_csv(csv_filename, sep='\t')
    return df

def _fix_topictokens():
    csv_filename = os.path.join(state.data_folder, state.basename, 'result_{}_topic_tokens.csv'.format(state.basename))

    topickeys = _read_topic_keys(state.data_folder, state.basename).set_index('topic_id')[['alpha']]
    topictokens = _read_result_model_sheet(csv_filename).set_index('topic_id')
    new_topictokens = pd.merge(topictokens, topickeys, how='inner', left_index=True, right_index=True)

    new_topictokens.to_csv(csv_filename, sep='\t')
    print('Topic Tokens Fixed!')
