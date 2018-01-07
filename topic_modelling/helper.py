# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import os
import gensim
from gensim.models.wrappers import ldamallet
from gensim.models.ldamodel import LdaModel
import glob
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import pyLDAvis
import pyLDAvis.gensim

import logging
logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s",  level=logging.INFO)

def load_mallet_lda_model(data_folder, basename):
    filename = os.path.join(data_folder, 'mallet_model_{}.gensim'.format(basename))
    mallet_lda = ldamallet.LdaMallet.load(filename, mmap=None)
    lda = ldamallet.malletmodel2ldamodel(mallet_lda)
    return lda

def load_gensim_lda_model(data_folder, basename):
    lda_filename = os.path.join(data_folder, 'gensim_model_{}.gensim'.format(basename))
    lda = LdaModel.load(lda_filename)
    return lda

def load_dictionary(data_folder):
    filename = os.path.join(data_folder, 'corpus.dict')
    dictionary = gensim.corpora.Dictionary.load(filename)
    return dictionary

def load_corpus(data_folder):
    # Note: Vocabulary is extracted from Corpus...????
    filename = os.path.join(data_folder, 'corpus.mm')
    corpus = gensim.corpora.MmCorpus(filename)
    return corpus

def load_result_excel_sheet(source_folder, basename, sheet):
    filename = os.path.join(source_folder, '{}/'.format(basename), 'result_{}.xlsx'.format(basename))
    with pd.ExcelFile(filename) as xls:
        df = pd.read_excel(xls, sheet)
    return df

def load_topic_tokens(source_folder, basename):
    return load_result_excel_sheet(source_folder, basename, 'topic_token_weights')

def load_mallet_document_topics(source_folder, basename, melt=False):
    filename = os.path.join(source_folder, '{}/'.format(basename), 'doctopics.txt')
    df = pd.read_table(filename, header=None, index_col=0)
    n_topics = len(df.columns) - 1
    df.columns = ['document_id'] + list(range(0, n_topics))
    if melt:
        df = pd.melt(df,
                     id_vars=['document_id'],
                     var_name="topic_id",
                     value_name="weight",
                     value_vars=list(range(0, n_topics)))
    return df

def load_document_topics(source_folder, basename, melt=False):
    df = load_result_excel_sheet(source_folder, basename, 'doc_topic_weights')
    n_topics = len(df.columns) - 3
    document_columns = [ 'document_id', 'document', 'year' ]
    topic_columns = list(range(0, n_topics))
    # FIXME: First topic should be 0 in Excel!!!!
    df.columns = document_columns + topic_columns
    if melt:
        df = pd.melt(df,
                     id_vars=document_columns,
                     var_name="topic_id",
                     value_name="weight",
                     value_vars=topic_columns)
    return df

def get_model_names(source_folder):
    return [ os.path.split(x)[1] for x in glob.glob(os.path.join(source_folder, '*')) ]

def compute_gensim_tfidf_bag_of_keywords(corpus):
    tfidf = gensim.models.TfidfModel(corpus, normalize=True)
    corpus_tfidf = tfidf[corpus]
    return corpus_tfidf

def compute_sklearn_tfidf(text_corpus, top_n, max_features=5000):
    """ return the top n feature names and idf scores of a tweets list """
    def documents(corpus):
        for document in corpus:
            yield ' '.join(document)

    tfidf_vectorizer = TfidfVectorizer(max_features=max_features)
    tfidf_vectorizer.fit_transform(documents(text_corpus))
    indices = np.argsort(tfidf_vectorizer.idf_)[::-1]
    features = tfidf_vectorizer.get_feature_names()
    top_feature_name = [features[i] for i in indices[:top_n]]
    top_feature_idf = tfidf_vectorizer.idf_[indices][:top_n]

    return top_feature_name, top_feature_idf

def transform_sklearn_to_gensim(corpus_vect):
    # transform sparse matrix into gensim corpus
    corpus_vect_gensim = gensim.matutils.Sparse2Corpus(corpus_vect, documents_columns=False)
    from gensim.corpora.dictionary import Dictionary
    dictionary = Dictionary.from_corpus(corpus_vect_gensim,
            id2word=dict((id, word) for word, id in corpus_vect.vocabulary_.items()))
    return corpus_vect_gensim, dictionary

def convert_to_pyLDAvis(lda, corpus, dictionary, R=100, mds='tsne', sort_topics=False, plot_opts={'xlab': 'PC1', 'ylab': 'PC2'}, target_folder=None, basename='pyldavis'):
    data = pyLDAvis.gensim.prepare(lda, corpus, dictionary, R=R, mds=mds, plot_opts=plot_opts, sort_topics=sort_topics)
    if target_folder is not None:
        pyLDAvis.save_json(data, os.path.join(target_folder, '{}.json'.format(basename)))
        pyLDAvis.save_html(data, os.path.join(target_folder, '{}.html'.format(basename)))
    return data
