# -*- coding: utf-8 -*-

import logging
import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim import models, matutils
from common.utility import FileUtility
from . import ModelUtility
from . import LdaMalletService

class ModelComputeHelper():

    @staticmethod
    def compute_gensim_tfidf_bag_of_keywords(corpus):
        tfidf = models.TfidfModel(corpus, normalize=True)
        corpus_tfidf = tfidf[corpus]
        return corpus_tfidf

    @staticmethod
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

    @staticmethod
    def transform_sklearn_to_gensim(corpus_vect):
        # transform sparse matrix into gensim corpus
        corpus_vect_gensim = matutils.Sparse2Corpus(corpus_vect, documents_columns=False)
        from gensim.corpora.dictionary import Dictionary
        dictionary = Dictionary.from_corpus(corpus_vect_gensim, id2word=dict((id, word) for word, id in corpus_vect.vocabulary_.items()))
        return corpus_vect_gensim, dictionary

def compute(corpus, documents, options):

    """
    Function for LDA training using gensim LDA or gensim's MALLET wrapper.
    Basic flow:
        - Convert (Sparv) corpus to MmCorpus
        - Store corpus, dictionary and document index 
        - Call selected LDA engine
        - Save model to disk

    'corpus' - a SparvTextCorpus (but can be any corpus)
    'documents' - a document id to document name mallet_path
    'options' - run time options for LDA execution 
    """

    data_folder = options['target_folder']
    basename = ModelUtility.create_basename(options)
    directory = os.path.join(options['target_folder'], basename + '\\')
    repository = FileUtility(directory).create(True)
    lda_options = options['lda_options']

    ModelUtility.store_corpus(data_folder, basename, corpus)
    ModelUtility.store_dictionary(data_folder, basename, corpus.dictionary)
    ModelUtility.store_document_index(data_folder, basename, documents)

    dictionary = ModelUtility.load_dictionary(data_folder, basename)
    mm = ModelUtility.load_corpus(data_folder, basename)

    if 'MALLET' in options.get('lda_engine', '').upper():

        lda_options.update({ 'prefix': repository.directory, "workers": 4, "optimize_interval": 10 })
        mallet_path = options.get('engine_path', '')
        model = LdaMalletService(mm, id2word=dictionary, mallet_path=mallet_path, **lda_options)

        # model.save(os.path.join(repository.directory, 'mallet_model_{}.gensim.gz'.format(basename)))

        ''' Convert to Gensim LDA model... '''
        lda = models.wrappers.ldamallet.malletmodel2ldamodel(model)

        ''' Compress files to save space... '''
        FileUtility.compress(model.ftopicwordweights())

    else:
        lda_options.update({ 'dtype': np.float64 })
        lda = models.LdaModel(corpus=mm, id2word=dictionary, **lda_options)

    ''' Persist gensim model to disk '''
    ModelUtility.store_gensim_lda_model(data_folder, basename, lda)


class LdaModelExtraDataCompiler():

    """
    Class that prepares and extracts various data from LDA model.
    Main purpose is to prepare data for Jupyter notebooks 
    """

    def get_document_topics(self, lda, mm, corpus_documents, num_words=200, minimum_probability=None):

        ''' Get document topic weights for all documents in corpus '''
        ''' Note!  minimum_probability=None filters less probable topics, set to 0 to retrieve all topcs'''

        df_doc_topics = pd.DataFrame(
            sum([ [ (i, x[0], x[1]) for x in topics ]
                for i, topics in enumerate(lda.get_document_topics(mm, minimum_probability=minimum_probability)) ], []),
            columns=[ 'document_id', 'topic_id', 'weight' ]
        )

        df = pd.merge(corpus_documents, df_doc_topics, how='inner', left_index=True, right_on='document_id')
        return df

    def get_topic_token_weight_toplist(self, lda, num_words=200):

        df_topic_weights = pd.DataFrame(
            [ (topic_id, token, weight)
                for topic_id, tokens in (lda.show_topics(lda.num_topics, num_words=num_words, formatted=False))
                    for token, weight in tokens if weight > 0.0 ],
            columns=['topic_id', 'token', 'weight']
        )
        return df_topic_weights

    def get_topic_token_overview(self, topic_token_weights):
        df = topic_token_weights.\
            groupby(['topic_id'])['token'].apply(' '.join).reset_index()
        return df

    def generate(self, data_folder, basename):

        lda = ModelUtility.load_gensim_lda_model(data_folder, basename)
        documents = ModelUtility.load_document_index(data_folder, basename)
        mm = ModelUtility.load_corpus(data_folder, basename)

        data = self._compile(lda, mm, documents)

        self._save(data, data_folder, basename)

    def _compile(self, lda, mm, documents):
        '''
        Prepare various data frames to be saved as sheets in an Excel file...
        '''

        dictionary = pd.DataFrame({
            'token_id': list(lda.dictionary.token2id.values()),
            'token': list(lda.dictionary.token2id.keys()),
            'dfs': list(lda.dictionary.dfs.values())
        }).set_index('token_id')[['token', 'dfs']]

        corpus_documents = pd.DataFrame(documents).set_index('document_id')
        doc_topic_weights = self.get_document_topics(lda, mm, corpus_documents, num_words=200, minimum_probability=0)
        topic_token_weights = self.get_topic_token_weight_toplist(lda, num_words=200)
        topic_overview = self.get_topic_token_overview(topic_token_weights)

        sheets = [
            (doc_topic_weights.reset_index(), 'doc_topic_weights'),
            (topic_overview, 'topic_tokens'),
            (topic_token_weights, 'topic_token_weights'),
            (corpus_documents, 'documents'),
            (dictionary, 'dictionary')
        ]

        return sheets

    def _save(self, data, data_folder, basename):
        """
        Saves data to Excel and as individual CSV files...
        'data' - a list of (df, name) pairs
        'data_folder' - data root folder
        'basename' - model identifier (used as sub folder name)
        """
        FileUtility.save_excel(data, os.path.join(data_folder, basename, 'result_' + basename + '.xlsx'))

        for df, dname in data:

            if not hasattr(df, 'to_csv'):
                continue

            filename = os.path.join(data_folder, basename, 'result_{}_{}.csv'.format(basename, dname))

            if not os.path.exists(filename):
                df.to_csv(filename, sep='\t')
