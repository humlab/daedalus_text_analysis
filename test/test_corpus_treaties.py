
import logging
import os
import re
import sys
import unittest
import zipfile

import gensim
import nltk
import pandas as pd

__cwd__ = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
__root_path__ = os.path.join(__cwd__, '..')
sys.path.append(__root_path__)

from common.utility import FileUtility, extend
from sparv_annotater import (RawTextCorpus, SparvCorpusReader,
                              SparvTextCorpus, TextCorpusReader,
                              ZipFileIterator)

logger = logging.getLogger(__name__)

DEFAULT_OPT = {
    "skip": False,
    "language": 'swedish',    
    "postags": '|NN|PM|',
    #"postags": '|NOUN|PROPN|', # For non-Swedish languages
    "chunk_size": None,
    "lowercase": True,
    "min_token_size": 3,
    'filter_stopwords': True,
    "lemmatize": True,
    'prune_at': 2000000,
    'root_folder': 'C:\\tmp\\',
    'doc_name_attrib_extractors': [('year', lambda x: int(re.search(r'(\d{4})', x).group(0)))]
}

# TODO: add test files, etc
# Test data
# test_source...

source = '..\\test_data\\treaties_corpus_pos_xml.zip'

options_list = [
    {
        "language": 'english',                
        'source': source,
        'postags': '|NOUN|PROPN|',
        'chunk_size': 1000,
        'lemmatize': True,                
        'doc_name_attrib_extractors': []
    },
]



def create_corpus(opt):

    language = opt.get("language", 'swedish')

    transformers = [
        (lambda tokens: [ x for x in tokens if any(map(lambda x: x.isalpha(), x)) ])
    ]

    if opt.get("filter_stopwords", False) is True:
        stopwords = nltk.corpus.stopwords.words(language)
        transformers.append(
            lambda tokens: [ x for x in tokens if x not in stopwords ]
        )

    min_token_size = opt.get("min_token_size", 3)
    if min_token_size > 0:
        transformers.append(
            lambda tokens: [ x for x in tokens if len(x) >= min_token_size ]
        )

    if opt.get("lowercase", True) is True:
        transformers.append(lambda _tokens: list(map(lambda y: y.lower(), _tokens)))
   
    postags = opt.get("postags", '') or ''
    stream = SparvCorpusReader(
        source=opt.get('source', []),
        transforms=transformers,
        postags="'{}'".format(postags),
        chunk_size=opt.get("chunk_size", None),
        lemmatize=opt.get("lemmatize", True)
    )

    corpus = SparvTextCorpus(stream, prune_at=opt.get("prune_at", 2000000))

    return corpus


def dummy():
    print("blä")

#for _options in options_list:

#            opt = extend(dict(DEFAULT_OPT), _options)
#            if opt.get('skip', False) is True:
#                continue

#            corpus = create_corpus(opt)
#            for x in corpus:
#                for token_id, token_count in x:
#                    print(corpus.dictionary[token_id], token_count)


# TODO: add tests
class CorpusTreatiesTests(unittest.TestCase):

    def setUp(self):
        pass

    def test_void(self):
        pass

    def test_test(self):
        dummy()
        pass

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()


## unit test cases
'''
1. Number of retrived documents must equal number of files in ZIP with extension XML
2. No words should be filtered out if postags is None, filter_stopwords is False and min_token_size is 0
   Test: Retrieved word count equals number of distinct words in corpus
3. If min_token_size > 0 then there cannot exist any word in corpus len < min_token_size
4. If filter_stopwords is true then [ x for x in corpus x in stopwords ] = [ ]
5. If postags = 
6. If lemma = False then non lemmatize words should exist (test på plural, testa på specifika ord)
7. token should not be interpukntering
8. Test lowercase
'''
