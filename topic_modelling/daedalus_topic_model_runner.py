
import re
import sys
import os

__cwd__ = os.path.abspath(__file__) if '__file__' in globals() else os.getcwd()

sys.path.append(os.path.join(__cwd__, '..'))

import topic_modelling
from sparv_annotater import SparvTextCorpus, SparvCorpusReader
from common.utility import extend
from . topic_modelling import convert_to_pyLDAvis

mallet_path = 'C:\\Usr\\mallet-2.0.8'

class DaedalusTopicModelRunner:

    def get_documents(self, corpus):
        documents = corpus.get_corpus_documents(
            attrib_extractors=[('year', lambda x: int(re.search('(\d{4})', x).group(0)))]
        )
        return documents

    def create_corpus(self, opt):

        stream = SparvCorpusReader(
            source=opt.get('source', []),
            postags="'{}'".format(opt.get("postags")),
            chunk_size=opt.get("chunk_size", None),
            lowercase=opt.get("lowercase", True),
            min_token_size=opt.get("min_token_size", 3),
            lemmatize=opt.get("lemmatize", True)
        )

        corpus = SparvTextCorpus(stream, prune_at=opt.get("prune_at", 2000000))
        documents = self.get_documents(corpus)

        return corpus, documents

    def compute(self, corpus, documents, options):

        topic_modelling.compute(corpus, documents, options=options)

DEFAULT_OPT = {
    "skip": False,
    "postags": '|NN|PM|',
    "chunk_size": None,
    "lowercase": True,
    "min_token_size": 3,
    "lemmatize": True,
    'lda_engine': 'LdaMallet',
    "lda_options": {
        "num_topics": 50,
        "iterations": 2000,
    },
    'prune_at': 2000000,
    'target_folder': '/tmp/'
}


if __name__ == "__main__":

    source = '../test/test_data/1987_article_pos_xml.zip'
    '''
    See https://spraakbanken.gu.se/korp/markup/msdtags.html for description of MSD-tag set
    '''
    options_list = [

        # { 'source': source, 'lda_engine': 'LdaModel', 'engine_lda_options': { "num_topics": 50, "iterations": 2000, 'chunksize': 10000, 'passes': 2 }},
        { 'source': source, 'lda_engine': 'LdaMallet', 'lda_options': { "num_topics": 50, "iterations": 2000 }, 'engine_path': mallet_path  },

        # { 'source': source, 'lda_engine': 'LdaMallet', 'lda_options': { "num_topics": 100, "iterations": 2000 }, 'engine_path': mallet_path  },
        # { 'source': source, 'lda_engine': 'LdaModel', 'lda_options': { "num_topics": 100, "iterations": 2000, 'chunksize': 10000, 'passes': 2  } },

        # { 'source': source, 'lda_engine': 'LdaModel', 'lda_options': { "num_topics": 150, "iterations": 2000, 'chunksize': 10000, 'passes': 2  } },
        # { 'source': source, 'lda_engine': 'LdaMallet', 'lda_options': { "num_topics": 150, "iterations": 2000 }, 'engine_path': mallet_path  }
        # { 'source': source, 'lda_engine': 'LdaModel', 'lda_options': { "num_topics": 50, "iterations": 999, 'chunksize': 10000, 'passes': 2 } }
    ]

    for _options in options_list:

        opt = extend(dict(DEFAULT_OPT), _options)

        if opt.get('skip', False) is True:
            continue

        runner = DaedalusTopicModelRunner()

        corpus, documents = runner.create_corpus(opt)

        runner.compute(corpus, documents, opt)

        data_folder = opt['target_folder']
        basename = topic_modelling.ModelUtility.create_basename(opt)
        
        topic_modelling.LdaModelExtraDataCompiler().generate(data_folder, basename)
        convert_to_pyLDAvis(data_folder, basename)
