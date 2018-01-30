# -*- coding: utf-8 -*-

import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')

import logging
import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim import models, matutils
from common.utility import FileUtility
from .model_utility import ModelUtility
from .lda_mallet_service import LdaMalletService

logger = logging.getLogger(__name__)

class ModelComputeHelper():

    @staticmethod
    def compute_gensim_tfidf_bag_of_keywords(corpus):
        tfidf = models.TfidfModel(corpus, normalize=True)
        corpus_tfidf = tfidf[corpus]
        return corpus_tfidf

    @staticmethod
    def compute_sklearn_tfidf(text_corpus, top_n, max_features=5000):
        """ return the top n feature names and idf scores of a tweets list """
        def documents(corpus):
            for document in corpus:
                yield ' '.join(document)

        tfidf_vectorizer = TfidfVectorizer(max_features=max_features)
        tfidf_vectorizer.fit_transform(documents(text_corpus))
        indices = np.argsort(tfidf_vectorizer.idf_)[::-1]
        features = tfidf_vectorizer.get_feature_names()
        top_feature_name = [features[i] for i in indices[:top_n]]
        top_feature_idf = tfidf_vectorizer.idf_[indices][:top_n]

        return top_feature_name, top_feature_idf

    @staticmethod
    def transform_sklearn_to_gensim(corpus_vect):
        # transform sparse matrix into gensim corpus
        corpus_vect_gensim = matutils.Sparse2Corpus(corpus_vect, documents_columns=False)
        from gensim.corpora.dictionary import Dictionary
        dictionary = Dictionary.from_corpus(corpus_vect_gensim, id2word=dict((id, word) for word, id in corpus_vect.vocabulary_.items()))
        return corpus_vect_gensim, dictionary

def compute(corpus, options):

    """
    Function for LDA training using gensim LDA or gensim's MALLET wrapper.
    Basic flow:
        - Convert (Sparv) corpus to MmCorpus
        - Store corpus, dictionary and document index
        - Call selected LDA engine
        - Save model to disk

    'corpus' - a SparvTextCorpus (but can be any corpus)
    'options' - run time options for LDA execution
    """

    data_folder = options['target_folder']
    basename = ModelUtility.create_basename(options)
    directory = os.path.join(options['target_folder'], basename + '\\')
    lda_options = options['lda_options']

    ModelUtility.store_corpus(data_folder, basename, corpus)
    ModelUtility.store_dictionary(data_folder, basename, corpus.dictionary)

    documents = corpus.get_corpus_documents(options.get('doc_name_attrib_extractors', []))
    ModelUtility.store_document_index(data_folder, basename, documents)

    dictionary = ModelUtility.load_dictionary(data_folder, basename)
    mm = ModelUtility.load_corpus(data_folder, basename)

    if 'MALLET' in options.get('lda_engine', '').upper():

        lda_options.update({ 'prefix': directory, "workers": 4, "optimize_interval": 10 })
        mallet_home = options.get('engine_path', '')
        mallet_model = LdaMalletService(mm, id2word=dictionary, default_mallet_home=mallet_home, **lda_options)
        FileUtility.compress(mallet_model.ftopicwordweights())

        '''
        Convert to Gensim LDA model...
        '''
        lda_model = models.wrappers.ldamallet.malletmodel2ldamodel(mallet_model)

        result_model = mallet_model

    else:
        lda_options.update({ 'dtype': np.float64 })
        lda_model = models.LdaModel(corpus=mm, id2word=dictionary, **lda_options)
        result_model = lda_model

    '''
    Persist gensim model to disk
    '''
    ModelUtility.store_gensim_lda_model(data_folder, basename, lda_model)

    return result_model
