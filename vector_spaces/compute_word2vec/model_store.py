
from gensim.models.word2vec import Word2Vec

class ModelStore:

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

    
    @staticmethod
    def create_basename(options):
        return 'w2v_model_{}_win_{}_dim_{}_iter_{}_mc_{}{}{}{}'.format(
            'cbow' if options['sg'] == 0 else 'skip_gram',
            options.get('window', 5),
            options.get('size', 100),
            options.get('iter', 5),
            options.get('min_count', 0),
            options.get('id',''),
            '_no_stopwords' if options.get('filter_stopwords') else '',
            '_bigrams' if options.get('bigram_transformer') else '') + '.dat'
