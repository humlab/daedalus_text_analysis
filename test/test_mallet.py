#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import unittest
import pandas as pd

from gensim import corpora, models
from topic_modelling import LdaMalletService, LdaModelExtraDataCompiler
from utility import generate_temp_filename
from common.utility import revdict, extend

mallet_path = 'C:\\Usr\\mallet-2.0.8'

test_texts = [
    ['människa', 'gränsyta', 'dator'],
    ['enkät', 'användare', 'dator', 'system', 'svar', 'tid'],
    ['mus', 'användare', 'gränsyta', 'system'],
    ['system', 'människa', 'system', 'mus'],
    ['användare', 'svar', 'tid'],
    ['träd'],
    ['graf', 'träd'],
    ['graf', 'barn', 'träd'],
    ['graf', 'barn', 'enkät'],
    ['roger', 'roger', 'roger', 'springer', 'hoppar', 'tränar']
]

test_dictionary = corpora.Dictionary(test_texts)
test_corpus = [ test_dictionary.doc2bow(text) for text in test_texts ]
test_corpus_documents = pd.DataFrame({
    'document': [ 'test_document_{}.txt'.format(i + 1) for i, x in enumerate(test_corpus) ],
    'length': [ len(x) for x in test_corpus ]
})

lda_options = dict(
    num_topics=2,
    iterations=10,
    prefix='/tmp/test/',
    workers=1,
    optimize_interval=0,
    passes=100
)

class LdaMalletTestCase(unittest.TestCase):

    def setUp(self):
        pass

    def test_void(self):
        pass

    def test_mallet_id2word(self):

        dictionary, corpus = test_dictionary, test_corpus

        print(test_dictionary.token2id)
        print(test_dictionary.id2token)
        model = LdaMalletService(corpus=corpus, id2word=dictionary, default_mallet_path=mallet_path, **lda_options)

        print(test_dictionary.id2token)
        print(dict(model.id2word))

    def test_mallet_to_gensim_lda_conv(self):

        dictionary, corpus = test_dictionary, test_corpus

        print(test_dictionary.token2id)
        print(test_dictionary.id2token)

        mallet_model = LdaMalletService(corpus=corpus, id2word=dictionary, default_mallet_path=mallet_path, **lda_options)
        gensim_model = models.wrappers.ldamallet.malletmodel2ldamodel(mallet_model)

        _ = gensim_model.id2word[0]

        self.assertDictEqual(dictionary.id2token, dict(mallet_model.id2word))
        self.assertDictEqual(dictionary.id2token, dict(gensim_model.id2word))

        lda_filename = generate_temp_filename('gensim_model.gensim.gz')
        gensim_model.save(lda_filename)

        stored_model = models.LdaModel.load(lda_filename)
        self.assertDictEqual(dictionary.id2token, dict(stored_model.id2word))
        self.assertDictEqual(dict(stored_model.id2word), dict(gensim_model.id2word))
        print(dict(stored_model.id2word))

    def test_df_dictionary(self):
        dictionary = test_dictionary
        _ = dictionary[0]
        df_dictionary = pd.DataFrame({
            'token_id': list(dictionary.token2id.values()),
            'token': list(dictionary.token2id.keys()),
            'dfs': list(dictionary.dfs.values())
        }).set_index('token_id')[['token', 'dfs']]
        self.assertDictEqual(dictionary.id2token, df_dictionary.token.to_dict())

    def test_doc_topic_weights(self):

        dictionary, corpus, corpus_documents = test_dictionary, test_corpus, test_corpus_documents

        mallet_model = LdaMalletService(corpus=corpus, id2word=dictionary, default_mallet_path=mallet_path, **lda_options)
        gensim_model = models.wrappers.ldamallet.malletmodel2ldamodel(mallet_model)

        extra = LdaModelExtraDataCompiler()

        document_topics = extra.get_document_topics(gensim_model, corpus, corpus_documents, num_words=200, minimum_probability=0)

        self.assertIsNotNone(document_topics)
        self.assertEqual(20, len(document_topics))

        self.assertAlmostEqual(len(corpus_documents) * 1.0, document_topics.weight.sum())
        self.assertAlmostEqual(1.0, document_topics.groupby('document').sum()['weight'].min())
        self.assertAlmostEqual(1.0, document_topics.groupby('document').sum()['weight'].max())
        self.assertAlmostEqual(1.0, document_topics.groupby('document').sum()['weight'].mean())

    def test_compute(self):

        source = './test__data/1987_article_xml.zip'
mallet_path = 
        options = [
            { 'lda_engine': 'LdaMallet', 'lda_options': { "num_topics": 50, "iterations": 2000 }, 'engine_path': 'C:\\Usr\\mallet-2.0.8'  }
        ]

        compute(source=source, options=options)
        
    #     df_topic_token_weights = ModelComputeHelper.get_topic_token_weight_toplist(lda, num_words=200)
    #     df_topic_overview = ModelComputeHelper.get_topic_overview(df_topic_token_weights)
    #     df_yearly_mean_topic_weights = ModelComputeHelper.get_yearly_mean_topic_weight(df_doc_topic_weights, df_topic_overview)
    #     df_dictionary = pd.DataFrame({ 'token': revdict(dictionary.token2id), 'dfs': dictionary.dfs }).reset_index().set_index('index')[['token', 'dfs']]

    # def xtest_xxx(self):
        
    #     df_doc_topic_weights = ModelComputeHelper.get_document_topics(lda, mm, df_corpus_document, num_words=200, minimum_probability=0)
    #     df_topic_token_weights = ModelComputeHelper.get_topic_token_weight_toplist(lda, num_words=200)
    #     df_topic_overview = ModelComputeHelper.get_topic_overview(df_topic_token_weights)
    #     df_yearly_mean_topic_weights = ModelComputeHelper.get_yearly_mean_topic_weight(df_doc_topic_weights, df_topic_overview)

    #     df_dictionary = pd.DataFrame({ 'token': revdict(dictionary.token2id), 'dfs': dictionary.dfs }).reset_index().set_index('index')[['token', 'dfs']]

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
