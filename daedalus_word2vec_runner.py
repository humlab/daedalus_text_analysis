
from vector_spaces import compute_word2vec

def extend(a, b):
    return a.update(b) or a

'''
sentences (iterable of iterables) – The sentences iterable can be simply a list of lists of tokens, but for larger corpora, consider an iterable that streams the sentences directly from disk/network. See BrownCorpus, Text8Corpus or LineSentence in word2vec module for such examples. If you don’t supply sentences, the model is left uninitialized – use if you plan to initialize it in some other way.
sg (int {1, 0}) – Defines the training algorithm. If 1, CBOW is used, otherwise, skip-gram is employed.
size (int) – Dimensionality of the feature vectors.
window (int) – The maximum distance between the current and predicted word within a sentence.
alpha (float) – The initial learning rate.
min_alpha (float) – Learning rate will linearly drop to min_alpha as training progresses.
seed (int) – Seed for the random number generator. Initial vectors for each word are seeded with a hash of the concatenation of word + str(seed). Note that for a fully deterministically-reproducible run, you must also limit the model to a single worker thread (workers=1), to eliminate ordering jitter from OS thread scheduling. (In Python 3, reproducibility between interpreter launches also requires use of the PYTHONHASHSEED environment variable to control hash randomization).
min_count (int) – Ignores all words with total frequency lower than this.
max_vocab_size (int) – Limits the RAM during vocabulary building; if there are more unique words than this, then prune the infrequent ones. Every 10 million word types need about 1GB of RAM. Set to None for no limit.
sample (float) – The threshold for configuring which higher-frequency words are randomly downsampled, useful range is (0, 1e-5).
workers (int) – Use these many worker threads to train the model (=faster training with multicore machines).
hs (int {1,0}) – If 1, hierarchical softmax will be used for model training. If set to 0, and negative is non-zero, negative sampling will be used.
negative (int) – If > 0, negative sampling will be used, the int for negative specifies how many “noise words” should be drawn (usually between 5-20). If set to 0, no negative sampling is used.
cbow_mean (int {1,0}) – If 0, use the sum of the context word vectors. If 1, use the mean, only applies when cbow is used.
hashfxn (function) – Hash function to use to randomly initialize weights, for increased training reproducibility.
iter (int) – Number of iterations (epochs) over the corpus.
trim_rule (function) – Vocabulary trimming rule, specifies whether certain words should remain in the vocabulary, be trimmed away, or handled using the default (discard if word count < min_count). Can be None (min_count will be used, look to keep_vocab_item()), or a callable that accepts parameters (word, count, min_count) and returns either gensim.utils.RULE_DISCARD, gensim.utils.RULE_KEEP or gensim.utils.RULE_DEFAULT. Note: The rule, if given, is only used to prune vocabulary during build_vocab() and is not stored as part of the model.
sorted_vocab (int {1,0}) – If 1, sort the vocabulary by descending frequency before assigning word indexes.
batch_words (int) – Target size (in words) for batches of examples passed to worker threads (and thus cython routines).(Larger batches will be passed if individual texts are longer than 10000 words, but the standard cython code truncates to that maximum.)
compute_loss (bool) – If True, computes and stores loss value which can be retrieved using model.get_latest_training_loss().
callbacks – List of callbacks that need to be executed/run at specific stages during training.
'''

default_run_opts = {
    'window': 5,
    'size': 400,
    'sg': 1,
    'min_count': 5,
    'iter': 20,
    'workers': 10,
}

default_opts = {
    'id': '',
    'filter_stopwords': True,
    'bigram_transformer': False,
    'segment_strategy': 'sentence',
    'segment_size': 0
}

if __name__ == "__main__":

    source_file = './data/segmented-yearly-volumes_articles.zip'
    # source_file = './data/test_articles.zip'
    output_path = './data/output'

    options_list = [
        { 'run_opts': { 'window': 2, 'size': 200 }, 'segment_strategy': 'sentence', 'bigram_transform': True },
        { 'run_opts': { 'window': 20, 'size': 200 }, 'segment_strategy': 'document', 'bigram_transform': True  },
        { 'run_opts': { 'window': 5, 'size': 200 }, 'segment_strategy': 'chunk', 'segment_size': 100, 'bigram_transform': True  }
    ]

    for options in options_list:

        options = extend(dict(default_opts), options)

        run_opts = extend(dict(default_run_opts), dict(options.get('run_opts', {})))

        if options.get('skip', False) is True:
            continue

        compute_word2vec(
            run_id=options.get('id', ''),
            source_path=options.get('input_path', source_file),
            output_path=output_path,
            filter_stopwords=options.get('filter_stopwords', True),
            segment_strategy=options.get('segment_strategy', 'sentence'),
            segment_size=options.get('segment_size', 0),
            bigram_transform=options.get('bigram_transform', False),
            run_opts=run_opts
        )
