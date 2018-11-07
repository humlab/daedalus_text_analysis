
import sys
import os
import nltk
import re

__cwd__ = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
__root_path__ = os.path.abspath(__cwd__)  # os.path.join(__cwd__, '.'))

sys.path.append(__root_path__)

import topic_modelling
import pandas as pd
import zipfile
from corpora.sparv_text_corpus import SparvTextCorpus
from corpora.corpus_source_reader import SparvCorpusSourceReader
from corpora.raw_text_corpus import RawTextCorpus
from corpora.zip_utility import ZipFileIterator
import common.utility as utility
import common.file_utility as file_utility

mallet_path = 'C:\\Usr\\mallet-2.0.8'

def save_documents(corpus, target_folder):
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

    if opt.get('corpus_type', 'sparv_xml') == 'sparv_xml':

        postags = opt.get("postags", '') or ''

        source = ZipFileIterator(opt.get('source', []), [ 'xml' ])

        stream = SparvCorpusSourceReader(
            source=source,
            transforms=transformers,
            postags="'{}'".format(postags),
            chunk_size=opt.get("chunk_size", None),
            lemmatize=opt.get("lemmatize", True)
        )

        corpus = SparvTextCorpus(stream, prune_at=opt.get("prune_at", 2000000))

    elif opt.get('corpus_type', '') == 'load_corpus_mm':

        corpus = None

    elif opt.get('corpus_type', '') == 'zipped_text_corpus':

        if not opt.get("postags") is None:
            raise AssertionError('Attribute postags not allowed for text corpus')

        if not opt.get("lemmatize", False) is True:
            raise AssertionError('Attribute lemmatize not allowed for text corpus')

        stream = ZipFileIterator(opt.get("source", []), [ 'txt' ])

        corpus = RawTextCorpus(
            stream,
            segment_strategy=opt.get("segment_strategy", "document"),
            segment_size=opt.get("chunk_size", 1000),
            transformers=transformers
        )

    else:
        print('Unknown corpus type')

    return corpus

def compute(corpus, store, options):

    return topic_modelling.compute(corpus, store=store, options=options)

DEFAULT_OPT = {
    "skip": False,
    'prefix': '',
    "language": 'swedish',
    "clear_target_folder": True,
    "corpus_type": "sparv_xml",
    "postags": '|NN|PM|',
    "chunk_size": None,
    "lowercase": True,
    "min_token_size": 3,
    'filter_stopwords': True,
    "lemmatize": True,
    "num_topics": [ 50 ],
    "engines": [ ],
    'prune_at': 2000000,
    'root_folder': 'C:\\tmp\\',
    'doc_name_attrib_extractors':[('year', lambda x: int(re.search(r'(\d{4})', x).group(0)))]
    # [('treaty_id', lambda x: int(re.search(r'^(\w{6})_', x).group(0)))]
}

to_sequence = lambda x: list(x if isinstance(x, (list, tuple)) else x)

def file_basename(filepath):
    return os.path.basename(os.path.splitext(filepath.replace('\\', '/'))[0])

if __name__ == "__main__":

    source = 'C:\\Users\\roma0050\\Documents\\Projects\\daedalus_text_analysis\\data\\daedalus_articles_pos_xml_1931-2017.zip'
    #source = 'H:\\TEMP\\SOU_1990.zip'

    '''
    See https://spraakbanken.gu.se/korp/markup/msdtags.html for description of MSD-tag set
    '''
    run_options = [
        {
            'prefix': '#filename#',
            'corpus_type': 'sparv_xml',  # 'load_corpus_mm', 'sparv_xml
            "postags": '|NN|PM|',
            'clear_target_folder': True,
            'source': source,
            'chunk_size': 1000,
            'num_topics': [ 50 ],
            'engines': [
                {
                    'engine_name': 'LdaModel',
                    'engine_option': {
                        'iterations': 2000,
                        'passes': 3
                    }
                },
                {
                    'engine_name': 'LdaMallet',
                    'engine_option': {
                        'iterations': 2000,
                        'passes': 3,
                        'engine_path': mallet_path
                    }
                }
            ]
        }
    ]

    for run_option in run_options:

        option = utility.extend(dict(DEFAULT_OPT), dict(run_option))

        if option.get('skip', False) is True:
            continue

        n_topics = to_sequence(run_option['num_topics'])

        engines = to_sequence(run_option['engines'])

        for engine in engines:

            print("engine: {}".format(engine['engine_name']))

            for n_topic in n_topics:

                option['engine_option'] = utility.extend(engine['engine_option'], dict(num_topics=n_topic))
                option['engine_name'] = engine['engine_name']

                if option['prefix'] == '#filename#':
                    option['prefix'] = file_basename(source)

                store = topic_modelling.ModelStore(option)

                file_utility.FileUtility(store.target_folder).create(option.get('clear_target_folder', True))

                corpus = create_corpus(option)
                model = compute(corpus, store=store, options=option)

                topic_modelling.generate_notebook_friendly_data(store, model)

                # topic_modelling.convert_to_pyLDAvis(data_folder, basename)
