
from vector_spaces import compute_word2vec

def extend(a, b):
    return a.update(b) or a

default_run_opts = {
    'window': 5,
    'size': 400,
    'sg': 1,
    'min_count': 5,
    'iter': 20,
    'workers': 10,
}

default_opts = {
    'run_id': '',
    'filter_stopwords': True,
    'bigram_transformer': False,
    'segment_strategy': 'sentence',
    'segment_size': 0
}

# TM: 50 100 150, Word2Vec-model:stats_sg_win20_dim100_iter20_min5_articles_1931-2014_no_stop_document

if __name__ == "__main__":

    source_path = './data/segmented-yearly-volumes_articles_1931-2017.zip'
    output_path = './data/output'

    options_list = [
        #  { 'run_opts': { 'window': 2, 'size': 200 }, 'segment_strategy': 'sentence', 'bigram_transform': True },
        #  { 'run_opts': { 'window': 20, 'size': 200 }, 'segment_strategy': 'document', 'bigram_transform': True  },
        #  { 'run_opts': { 'window': 5, 'size': 200 }, 'segment_strategy': 'chunk', 'segment_size': 100, 'bigram_transform': True  },

        { 'run_id': 'daedalus_1931-2017', 'run_opts': { 'window': 20, 'size': 100 }, 'segment_strategy': 'document', 'bigram_transform': False  },

    ]

    for options in options_list:

        options = extend(dict(default_opts), options)

        options['run_opts'] = extend(dict(default_run_opts), dict(options.get('run_opts', {})))

        if options.get('skip', False) is True:
            continue

        options = extend(options, dict(
            source_path=source_path,
            output_path=output_path,
            run_opts=extend(dict(default_run_opts), dict(options.get('run_opts', {})))
        ))

        compute_word2vec(options)
