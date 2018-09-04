import gzip
import shutil
import os
import gensim.models
import nltk.corpus
import logging
import pandas as pd
from common import FileUtility
from . model_store import ModelStore
from sparv_annotater import RawTextCorpus, ZipFileIterator

logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO)

extend = lambda a, b: a.update(b) or a

def compress(filename):
    with open(filename, 'rb') as f_in:
        with gzip.open(filename + '.gz', 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

class Word2Vectorizer(object):

    def process(self, source_stream, filter_stopwords=True, segment_strategy='sentence', segment_size=0, bigram_transform=False, **run_opts):

        transformers = [
            (lambda tokens: [ x for x in tokens if any(map(lambda x: x.isalpha(), x)) ])
        ]

        if filter_stopwords is True:
            stopwords = nltk.corpus.stopwords.words('swedish')
            transformers.append(
                lambda tokens: [ x for x in tokens if x not in stopwords ]
            )

        if bigram_transform is True:
            train_corpus = RawTextCorpus(source_stream, segment_strategy='sentence', transformers=[])
            phrases = gensim.models.phrases.Phrases(train_corpus)
            bigram = gensim.models.phrases.Phraser(phrases)
            transformers.append(
                lambda tokens: bigram[tokens]
            )

        corpus = RawTextCorpus(
            source_stream,
            segment_strategy=segment_strategy,
            segment_size=segment_size,
            transformers=transformers
        )

        model = gensim.models.word2vec.Word2Vec(corpus, **run_opts)

        return model, corpus.stats

def compute_word2vec(opts):

    store = ModelStore(opts)

    if not store.source_path.endswith('zip'):
        raise Exception('Only source implemented is a ZIP-file that contains TXT-files')

    source_stream = ZipFileIterator(store.source_path, [ 'txt' ])

    vectorizer = Word2Vectorizer()

    model, corpus_stats = vectorizer.process(
        source_stream,
        filter_stopwords=opts.get('filter_stopwords'),
        segment_strategy=opts.get('segment_strategy'),
        segment_size=opts.get('segment_size'),
        bigram_transform=opts.get('bigram_transform'),
        **opts.get('run_opts')
    )

    FileUtility(store.target_folder).create(True)

    store.store_model(model, corpus_stats)
