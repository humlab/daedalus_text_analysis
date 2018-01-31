# -*- coding: utf-8 -*-
import os
from gensim.models import Word2Vec
import pandas as pd
import numpy as np
from scipy import spatial
import glob
import re

import logging
logger = logging.getLogger(__name__)

class WordVectorUtility:

    @staticmethod
    def get_model_names(source_folder):
        return [ os.path.split(x)[1] for x in glob.glob(os.path.join(source_folder, '*.dat')) ]

    @staticmethod
    def load_model_vector(filename):
        model = Word2Vec.load(filename)
        word_vectors = model.wv
        del model
        return word_vectors

    @staticmethod
    def create_X_m_space_matrix(word_vectors, words):
        index2word = [ x for x in words if x in word_vectors.vocab ]
        dim = word_vectors.syn0.shape[1]
        X_m_space = np.ndarray(shape=(len(index2word), dim), dtype='float64')
        for i in range(0, len(index2word)):
            X_m_space[i] = word_vectors[index2word[i]]
        return X_m_space, index2word

    @staticmethod
    def split_word_expression(wexpr):
        wexpr = wexpr.lower().replace(' ', '')
        positives = re.findall(r"(?:^|(?<![-\w]))([\w]+)", wexpr)
        negatives = re.findall(r"-([\w]+)", wexpr)
        return {
            'positives': [ x for x in positives if x not in negatives ],
            'negatives': [ x for x in negatives if x not in positives ]
        }

    @staticmethod
    def compute_most_similar_expression(word_vectors, wexpr):
        try:
            options = WordVectorUtility.split_word_expression(wexpr)
            result = word_vectors.most_similar(positive=options['positives'], negative=options['negatives'])
            return result, options or dict(positives=[], negatives=[])
        except Exception as ex:
            logger.error(str(ex))
            return None, None

    @staticmethod
    def compute_similarity_to_anthologies(word_vectors, scale_x_pair, scale_y_pair, word_list):

        scale_x = word_vectors[scale_x_pair[0]] - word_vectors[scale_x_pair[1]]
        scale_y = word_vectors[scale_y_pair[0]] - word_vectors[scale_y_pair[1]]

        word_x_similarity = [1 - spatial.distance.cosine(scale_x, word_vectors[x]) for x in word_list ]
        word_y_similarity = [1 - spatial.distance.cosine(scale_y, word_vectors[x]) for x in word_list ]

        df = pd.DataFrame({ 'word': word_list, 'x': word_x_similarity, 'y': word_y_similarity })

        return df

    @staticmethod
    def compute_similarity_to_single_words(word_vectors, word_x, word_y, word_list):

        word_x_similarity = [ word_vectors.similarity(x, word_x) for x in word_list ]
        word_y_similarity = [ word_vectors.similarity(x, word_y) for x in word_list ]

        df = pd.DataFrame({ 'word': word_list, 'x': word_x_similarity, 'y': word_y_similarity })

        return df

    @staticmethod
    def compute_similarity_to_words(word_vectors, x, y, word_list):

        if type(x) == 'tuple' and type(y) == 'tuple':
            return WordVectorUtility.compute_similarity_to_anthologies(word_vectors, x, y, word_list)

        if type(x) == 'str' and type(y) == 'str':
            return WordVectorUtility.compute_similarity_to_single_words(word_vectors, x, y, word_list)
        logger.error('Error: x and y must be wither two strings or two pair of strings')
        return None

    @staticmethod
    def seed_word_toplist(word_vectors, seed_word, topn=100):
        return [ seed_word ] + [ z[0] for z in word_vectors.most_similar_cosmul(seed_word, topn=topn) ]


