#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
import logging
import unittest
import tempfile
from sparv_annotater import SparvCorpusReader, SparvTextCorpus
from topic_modelling import ModelUtility
# from utility import join_test_data_path, generate_temp_filename
# from gensim import corpora
import logging

logger = logging.getLogger(__name__)

class ModelUtilityTestCase(unittest.TestCase):

    def setUp(self):
        pass

    def test_void(self):
        pass

    def test_can_store_document_index(self):

        xml_data = '''
            <corpus><paragraph>
            <sentence id="xxx">
            <w pos="NN"  lemma="|">Humlab</w>
            <w pos="VB"  lemma="|vara|">är</w>
            <w pos="DT"  lemma="|en|">ett</w>
            <w pos="NN"  lemma="|exempel|">exempel</w>
            <w pos="PP"  lemma="|på|">på</w>
            <w pos="DT"  lemma="|en|">ett</w>
            <w pos="NN"  lemma="|arbetsenhet|">arbetsenhet</w>
            <w pos="HP"  lemma="|">som</w>
            <w pos="JJ"  lemma="|digital|">digital</w>
            <w pos="VB"  lemma="|utför|">utför</w>
            <w pos="NN"  lemma="|forskning|">forskning</w>
            <w pos="MAD" lemma="|" >.</w>
            </sentence>
            <sentence id="xxx">
            <w pos="NN"  lemma="|">Humlab</w>
            <w pos="VB"  lemma="|vara|">är</w>
            <w pos="DT"  lemma="|en|">en</w>
            <w pos="NN"  lemma="|arbetsenhet|">arbetsenhet</w>
            <w pos="KN"  lemma="|och|">och</w>
            <w pos="AB"  lemma="|inte|">inte</w>
            <w pos="DT"  lemma="|en|">en</w>
            <w pos="NN"  lemma="|centrumbildning|">centrumbildning</w>
            <w pos="MAD" lemma="|" >.</w>
            </sentence>
            </paragraph></corpus>
            '''

        source = [ ('test_1987_01.xml', xml_data), ('test_1987_02.xml', xml_data), ('test_1987_03.xml', xml_data) ]

        
        stream = SparvCorpusReader(source=source, postags="'|NN|'", chunk_size=None, lowercase=True, min_token_size=3, lemmatize=True)

        corpus = SparvTextCorpus(stream=stream)

        self.assertIsNotNone(corpus)

        doc_name_attrib_extractors = [ ('year', lambda x: int(re.search(r'(\d{4})', x).group(0))) ]
        documents = corpus.get_corpus_documents(doc_name_attrib_extractors)

        self.assertIsNotNone(documents)
        self.assertIsNotNone(documents['document'])
        self.assertListEqual([ 'test_1987_01_01.txt', 'test_1987_02_01.txt', 'test_1987_03_01.txt' ], list(documents['document']))
        self.assertListEqual([ 0, 1, 2 ], list(documents['document_id']))

        with tempfile.TemporaryDirectory() as data_folder:
            df_stored = ModelUtility.store_document_index(data_folder, '', documents)
            df_loaded = ModelUtility.load_document_index(data_folder, '')
            self.assertTrue(df_stored.equals(df_loaded))

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
