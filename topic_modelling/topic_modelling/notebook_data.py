# -*- coding: utf-8 -*-

import logging
import os
import pandas as pd
from common.utility import FileUtility
from .model_utility import ModelUtility

class NotebookDataGenerator():

    """
    Class that prepares and extracts various data from LDA model.
    Main purpose is to prepare data for Jupyter notebooks
    """

    def _compile_dictionary(self, lda):
        token_ids, tokens = list(zip(*lda.id2word.items()))

        dictionary = pd.DataFrame({
            'token_id': token_ids,
            'token': tokens,
            'dfs': list(lda.id2word.dfs.values())
        }).set_index('token_id')[['token', 'dfs']]
        return dictionary

    def _compile_document_topics(self, lda, mm, corpus_documents, minimum_probability=None):

        '''
        Get document topic weights for all documents in corpus
        Note!  minimum_probability=None filters less probable topics, set to 0 to retrieve all topcs

        If gensim model then use 'get_document_topics', else 'load_document_topics' for mallet model
        '''
        data_iter = lda.get_document_topics(mm, minimum_probability=minimum_probability)\
            if hasattr(lda, 'get_document_topics')\
            else lda.load_document_topics()

        df_doc_topics = pd.DataFrame(
            sum([ [ (i, x[0], x[1]) for x in topics ] for i, topics in enumerate(data_iter) ], []),
            columns=[ 'document_id', 'topic_id', 'weight' ]
        )

        df = pd.merge(corpus_documents, df_doc_topics, how='inner', left_index=True, right_on='document_id')
        return df

    def _compile_topic_token_weights(self, lda, dictionary, num_words=200):

        df_topic_weights = pd.DataFrame(
            [ (topic_id, token, weight)
                for topic_id, tokens in (lda.show_topics(lda.num_topics, num_words=num_words, formatted=False))
                    for token, weight in tokens if weight > 0.0 ],
            columns=['topic_id', 'token', 'weight']
        )
        df = pd.merge(
            df_topic_weights,
            dictionary.reset_index().set_index('token'),
            how='inner',
            left_on='token',
            right_index=True
        )
        return df[['topic_id', 'token_id', 'token', 'weight']]

    def _compile_topic_token_overview(self, topic_token_weights, alpha, n_words=200):
        """
        Group by topic_id and concatenate n_words words within group sorted by weight descending.
        There must be a better way of doing this...
        """

        df = topic_token_weights.groupby('topic_id')\
            .apply(lambda x: sorted(list(zip(x["token"], x["weight"])), key=lambda z: z[1], reverse=True))\
            .apply(lambda x: ' '.join([z[0] for z in x][:n_words])).reset_index()
        df['alpha'] = df.topic_id.apply(lambda topic_id: alpha[topic_id])
        df.columns = ['topic_id', 'tokens', 'alpha']

        return df.set_index('topic_id')

    def generate(self, lda, data_folder, basename):

        if lda is None:
            lda = ModelUtility.load_gensim_lda_model(data_folder, basename)

        topic_keys = self._read_topic_keys(data_folder, basename)
        document_index = ModelUtility.load_document_index(data_folder, basename)
        mm = ModelUtility.load_corpus(data_folder, basename)

        data = self._compile(lda, mm, document_index, topic_keys)

        self._save(data, data_folder, basename)

    def _read_topic_keys(self, data_folder, basename):
        filename = os.path.join(data_folder, basename, 'topickeys.txt')
        if os.path.isfile(filename):
            df = pd.read_csv(filename, sep='\t', header=None, names=['topic_id', 'alpha', 'tokens'], decimal=b',')
            return df
        return None

    def _compile(self, lda, mm, document_index, topic_keys):
        '''
        Prepare various data frames to be saved as sheets in an Excel file...
        '''
        dictionary = self._compile_dictionary(lda)
        doc_topic_weights = self._compile_document_topics(lda, mm, document_index, minimum_probability=0)
        topic_token_weights = self._compile_topic_token_weights(lda, dictionary, num_words=200)

        alpha = lda.alpha if topic_keys is None else topic_keys.alpha
        topic_overview = self._compile_topic_token_overview(topic_token_weights, alpha)

        sheets = [
            (doc_topic_weights, 'doc_topic_weights'),
            (topic_overview, 'topic_tokens'),
            (topic_token_weights, 'topic_token_weights'),
            (document_index, 'documents'),
            (dictionary, 'dictionary')
        ]
        if topic_keys is not None:
            sheets.append((topic_keys, 'topickeys'))

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

def generate_notebook_friendly_data(lda_model, data_folder, basename):
    NotebookDataGenerator().generate(lda_model, data_folder, basename)
