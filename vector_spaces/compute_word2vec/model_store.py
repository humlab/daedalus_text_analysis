
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

