
import re
import sys
import os
import nltk
import gensim

__cwd__ = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
__root_path__ = os.path.abspath(__cwd__)  # os.path.join(__cwd__, '.'))

sys.path.append(__root_path__)

import topic_modelling
import pandas as pd
import zipfile
from sparv_annotater import SparvTextCorpus, SparvCorpusReader, RawTextCorpus, ZipFileIterator
from common.utility import extend, FileUtility

mallet_path = 'C:\\Usr\\mallet-2.0.8'

class DaedalusTopicModelRunner:

    def save_documents(self, corpus, target_folder):
        result_zip_name = os.path.join(target_folder, 'documents.zip')
        with zipfile.ZipFile(result_zip_name, 'w', zipfile.ZIP_DEFLATED) as zf:
            for i, document in enumerate(corpus):
                document_name = '{}.csv'.format(corpus.corpus_documents[i].split('.')[0])
                token_ids, token_counts = list(zip(*document))
                df = pd.DataFrame(dict(token_id=token_ids, token_count=token_counts))
                df['token'] = df.token_id.apply(lambda x: corpus.dictionary[x])
                df.set_index('token_id', inplace=True)
                content = u'\ufeff' + df.to_csv(path_or_buf=None, sep=';', header=True, encoding='utf-8')
                zf.writestr(document_name, content, compress_type=zipfile.ZIP_DEFLATED)

    def create_corpus(self, opt):

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

        if opt.get('corpus_type', 'sparv_xml') == 'sparv_xml':
            postags = opt.get("postags", '') or ''
            stream = SparvCorpusReader(
                source=opt.get('source', []),
                transforms=transformers,
                postags="'{}'".format(postags),
                chunk_size=opt.get("chunk_size", None),
                lemmatize=opt.get("lemmatize", True)
            )

            corpus = SparvTextCorpus(stream, prune_at=opt.get("prune_at", 2000000))

        elif opt.get('corpus_type', '') == 'load_corpus_mm':

            corpus = None

        else:

            if not opt.get("postags") is None:
                raise AssertionError('Attribute postags not allowed for text corpus')

            if not opt.get("lemmatize", False) is True:
                raise AssertionError('Attribute lemmatize not allowed for text corpus')

            stream = ZipFileIterator(source, [ 'txt' ])

            corpus = RawTextCorpus(
                stream,
                segment_strategy=opt.get("segment_strategy", "document"),
                segment_size=opt.get("chunk_size", 1000),
                transformers=transformers
            )

        return corpus

    def compute(self, corpus, store, options):

        return topic_modelling.compute(corpus, store=store, options=options)

DEFAULT_OPT = {
    "skip": False,
    "language": 'swedish',
    "clear_target_folder": True,
    "corpus_type": "sparv_xml",
    "postags": '|NN|PM|',
    "chunk_size": None,
    "lowercase": True,
    "min_token_size": 3,
    'filter_stopwords': True,
    "lemmatize": True,
    'lda_engine': 'LdaMallet',
    "lda_options": {
        "num_topics": 50,
        "iterations": 2000,
    },
    'prune_at': 2000000,
    'root_folder': 'C:\\tmp\\',
    'doc_name_attrib_extractors': [('year', lambda x: int(re.search(r'(\d{4})', x).group(0)))]
}

if __name__ == "__main__":

    #  source = 'C:\\Users\\roma0050\\Documents\\Projects\\daedalus_text_analysis\\data\\daedalus_articles_pos_xml.zip'
    source = 'H:\\Temp\\SOU_1945-1989.zip'
    #  source = 'H:\\Temp\\1945_10.zip'

    '''
    See https://spraakbanken.gu.se/korp/markup/msdtags.html for description of MSD-tag set
    '''

    for n_topics in [300]:

        options_list = [
            {
                'corpus_type': 'sparv_xml',
                #  'corpus_type': 'load_corpus_mm',
                'clear_target_folder': False,
                'source': source,
                'postags': None,
                'chunk_size': 1000,
                'lemmatize': False,
                'lda_engine': 'LdaMallet',
                'lda_options': {
                    "num_topics": n_topics,
                    "iterations": 2000
                },
                'engine_path': mallet_path
            },

        ]

        for _options in options_list:

            opt = extend(dict(DEFAULT_OPT), _options)

            if opt.get('skip', False) is True:
                continue

            store = topic_modelling.ModelStore(opt)

            FileUtility(store.target_folder).create(opt.get('clear_target_folder', True))

            runner = DaedalusTopicModelRunner()
            corpus = runner.create_corpus(opt)
            model = runner.compute(corpus, store=store, options=opt)

            topic_modelling.generate_notebook_friendly_data(store)

            # topic_modelling.convert_to_pyLDAvis(data_folder, basename)
