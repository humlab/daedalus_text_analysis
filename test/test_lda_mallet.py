#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import unittest
from gensim import corpora
from topic_modelling import compute
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
    ['graf', 'barn', 'enkät']
]

test_dictionary = corpora.Dictionary(test_texts)
test_corpus = [ test_dictionary.doc2bow(text) for text in test_texts ]

class LdaMalletTestCase(unittest.TestCase):

    def setUp(self):
        pass

    def test_mallet(self):
        dictionary = test_dictionary
        corpus = test_corpus
        lda_options = dict(num_topics=2, iterations=10, prefix='', workers=1, optimize_interval=0, passes=100)
        _ = LdaMalletService(corpus, id2word=dictionary, mallet_path=mallet_path, **lda_options)

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
