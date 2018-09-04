import os
import time
from gensim.models.word2vec import Word2Vec
import pandas as pd
join = os.path.join

class ModelStore:

    def __init__(self, opts):
        self.basename = self.create_basename(opts)
        self.output_path = opts.get('output_path', '.')
        self.source_path = opts.get('source_path')
        self.target_folder = join(self.output_path, self.basename + '\\')
        self.model_filename = join(self.target_folder, '{}.dat'.format(self.basename))
        self.model_tsv_filename = join(self.target_folder, 'vector_{}.tsv'.format(self.basename))
        self.stats_filename = join(self.target_folder, 'stats_{}.tsv'.format(self.basename))

    def create_basename(self, opts):
        ts = time.strftime("%Y%m%d")
        run_opts = opts.get('run_opts')
        filter_stopwords = opts.get('filter_stopwords', False)
        bigram_transform = opts.get('bigram_transform', False)
        segment_strategy = opts.get('segment_strategy')
        segment_size = opts.get('segment_size')
        run_id = opts.get('run_id')
        return '{}_{}_{}_w{}_d{}_i{}_min{}{}{}{}'.format(
            ts, run_id,
            'cbow' if run_opts['sg'] == 0 else 'sg',
            run_opts.get('window', 5),
            run_opts.get('size', 100),
            run_opts.get('iter', 5),
            run_opts.get('min_count', 0),
            '_nostop' if filter_stopwords else '',
            '_bg' if bigram_transform else '',
            '_{}{}'.format(segment_strategy, str(segment_size) if segment_strategy == 'chunk' else '')
        )

    def store_model(self, model, corpus_stats=None):
        model.save(self.model_filename)
        model.wv.save_word2vec_format(self.model_tsv_filename)
        # W2V_TensorFlow().convert_file(model_filename, dimension=options['size'])
        if corpus_stats is not None:
            stats = pd.DataFrame(corpus_stats, columns=['filename', 'total_tokens', 'tokens'])
            stats.to_csv(self.stats_filename, sep='\t')

    @staticmethod
    def load_model(filename):
        model = Word2Vec.load(filename)
        return model

    @staticmethod
    def load_model_vector(filename):
        model = ModelStore.load_model(filename)
        word_vectors = model.wv
        del model
        return word_vectors

