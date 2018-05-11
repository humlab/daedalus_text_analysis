import gzip
import shutil
import os
import gensim.models
import nltk.corpus
import logging
import pandas as pd
from common import FileUtility
from sparv_annotater import RawTextCorpus, ZipFileIterator

logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO)

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


def create_basename(run_id, filter_stopwords, bigram_transform, segment_strategy, segment_size, run_opts):
    return '{}_win{}_dim{}_iter{}_min{}{}{}{}{}'.format(
        'cbow' if run_opts['sg'] == 0 else 'sg',
        run_opts.get('window', 5),
        run_opts.get('size', 100),
        run_opts.get('iter', 5),
        run_opts.get('min_count', 0),
        '_nostop' if filter_stopwords else '',
        '_bg' if bigram_transform else '',
        '_{}{}'.format(segment_strategy, str(segment_size) if segment_strategy == 'chunk' else ''),
        run_id
    )

def compute_word2vec(
    run_id,
    source_path,
    output_path,
    filter_stopwords=True,
    segment_strategy='sentence',
    segment_size=0,
    bigram_transform=False,
    run_opts=None
):

    assert run_opts is not None

    basename = create_basename(run_id, filter_stopwords, bigram_transform, segment_strategy, segment_size, run_opts)
    directory = os.path.join(output_path, basename + '\\')

    if not source_path.endswith('zip'):
        raise Exception('Only source implemented is a ZIP-file that contains TXT-files')

    source_stream = ZipFileIterator(source_path, [ 'txt' ])

    vectorizer = Word2Vectorizer()

    model, corpus_stats = vectorizer.process(
        source_stream,
        filter_stopwords=filter_stopwords,
        segment_strategy=segment_strategy,
        segment_size=segment_size,
        bigram_transform=bigram_transform,
        **run_opts
    )

    FileUtility(directory).create(True)

    model_filename = os.path.join(directory, '{}.dat'.format(basename))
    model.save(model_filename)

    model_tsv_filename = os.path.join(directory, 'vector_{}.tsv'.format(basename))
    model.wv.save_word2vec_format(model_tsv_filename)

    # W2V_TensorFlow().convert_file(model_filename, dimension=options['size'])
    stats = pd.DataFrame(corpus_stats, columns=['filename', 'total_tokens', 'tokens'])
    stats.to_csv(os.path.join(directory, 'stats_{}.tsv'.format(basename)), sep='\t')
