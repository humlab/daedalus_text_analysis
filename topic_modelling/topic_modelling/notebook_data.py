# -*- coding: utf-8 -*-

import logging
import os
import pandas as pd
from common.utility import FileUtility
from gensim.models.ldamodel import LdaModel
from gensim.corpora import MmCorpus

join = os.path.join
logger = logging.getLogger('NotebookDataGenerator')
logger.setLevel(logging.INFO)
'''
OK model.id2word
model.get_document_topics
  OK model.load_document_topics()
OK model.show_topics(model.num_topics, num_words=num_words, formatted=False)
OK model.num_topics
OK model.alpha
self.store.load_gensim_lda_model()
'''
class NotebookDataGenerator():

    def __init__(self, store):
        self.store = store

    """
    Class that prepares and extracts various data from LDA model.
    Main purpose is to prepare data for Jupyter notebooks
    """

    def _compile_dictionary(self, lda):
        logger.info('Compiling dictionary...')
        token_ids, tokens = list(zip(*lda.id2word.items()))
        dfs = lda.id2word.dfs.values() if lda.id2word.dfs is not None else [0] * len(tokens)
        dictionary = pd.DataFrame({
            'token_id': token_ids,
            'token': tokens,
            'dfs': list(dfs)
        }).set_index('token_id')[['token', 'dfs']]
        return dictionary

    def __compile_document_topics_iter(self, lda, mm, minimum_probability):

        data_iter = lda.get_document_topics(mm, minimum_probability=minimum_probability)\
            if hasattr(lda, 'get_document_topics')\
            else lda.load_document_topics()

        for i, topics in enumerate(data_iter):
            for x in topics:
                yield (i, x[0], x[1])

    def _compile_document_topics(self, lda, mm, corpus_documents, minimum_probability=0.001):

        '''
        Get document topic weights for all documents in corpus
        Note!  minimum_probability=None filters less probable topics, set to 0 to retrieve all topcs

        If gensim model then use 'get_document_topics', else 'load_document_topics' for mallet model
        '''
        logger.info('Compiling document topics...')
        logger.info('  Creating data iterator...')
        data = self.__compile_document_topics_iter(lda, mm, minimum_probability)
        logger.info('  Creating frame from iterator...')
        df_doc_topics = pd.DataFrame(data, columns=[ 'document_id', 'topic_id', 'weight' ]).set_index('document_id')
        logger.info('  Merging data...')
        df = pd.merge(corpus_documents, df_doc_topics, how='inner', left_index=True, right_index=True)
        return df

    def _compile_topic_token_weights(self, lda, dictionary, num_words=200):
        logger.info('Compiling topic-tokens weights...')

        df_topic_weights = pd.DataFrame(
            [ (topic_id, token, weight)
                for topic_id, tokens in (lda.show_topics(lda.num_topics, num_words=num_words, formatted=False))
                    for token, weight in tokens if weight > 0.0 ],
            columns=['topic_id', 'token', 'weight']
        )

        df = pd.merge(
            df_topic_weights.set_index('token'),
            dictionary.reset_index().set_index('token'),
            how='inner',
            left_index=True,
            right_index=True
        )
        return df.reset_index()[['topic_id', 'token_id', 'token', 'weight']]

    def _compile_topic_token_overview(self, topic_token_weights, alpha, n_words=200):
        """
        Group by topic_id and concatenate n_words words within group sorted by weight descending.
        There must be a better way of doing this...
        """
        logger.info('Compiling topic-tokens overview...')

        df = topic_token_weights.groupby('topic_id')\
            .apply(lambda x: sorted(list(zip(x["token"], x["weight"])), key=lambda z: z[1], reverse=True))\
            .apply(lambda x: ' '.join([z[0] for z in x][:n_words])).reset_index()
        df['alpha'] = df.topic_id.apply(lambda topic_id: alpha[topic_id])
        df.columns = ['topic_id', 'tokens', 'alpha']

        return df.set_index('topic_id')

    def generate(self, lda):

        # if lda is None:
        #    logger.info('Loading LDA model...')
        #    lda = self.store.load_gensim_lda_model()

        topic_keys = self.store.load_mallet_topic_keys()

        logger.info('Loading document index...')
        document_index = self.store.load_document_index()

        logger.info('Loading MM corpus...')
        mm = self.store.load_corpus()

        data = self._compile(lda, mm, document_index, topic_keys)

        self._save(data)

    def _compile(self, lda, mm, document_index, topic_keys):
        '''
        Prepare various data frames to be saved as sheets in an Excel file...
        '''
        dictionary = self._compile_dictionary(lda)
        doc_topic_weights = self._compile_document_topics(lda, mm, document_index, minimum_probability=0.001)
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

    def _save(self, data):
        """
        Saves data to Excel and as individual CSV files...
        'data' - a list of (df, name) pairs
        """
        logger.info('Saving excel...')
        FileUtility.save_excel(data, self.store.result_excel_filename)

        for df, dname in data:

            if not hasattr(df, 'to_csv'):
                continue

            filename = join(self.store.target_folder, 'result_{}_{}.csv'.format(self.store.basename, dname))

            if not os.path.exists(filename):
                logger.info('Extracting sheet {} to CSV...'.format(dname))
                df.to_csv(filename, sep='\t')


def generate_notebook_friendly_data(store, lda):
    NotebookDataGenerator(store).generate(lda)
