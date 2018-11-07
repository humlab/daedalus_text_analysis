# -*- coding: utf-8 -*-

import warnings

warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')

import logging
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim import models, matutils
from gensim.corpora import Dictionary
from common.file_utility import FileUtility
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
        dictionary = Dictionary.from_corpus(corpus_vect_gensim, id2word=dict((id, word) for word, id in corpus_vect.vocabulary_.items()))
        return corpus_vect_gensim, dictionary

def compute(corpus, store, options):

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

    if corpus is not None:

        # Convert corpus to MM format. Otherwise assume MM-corpus is on disk
        store.store_corpus(corpus)
        store.store_dictionary(corpus.dictionary)

        documents = corpus.get_corpus_documents(options.get('doc_name_attrib_extractors', []))
        store.store_document_index(documents)

    dictionary = store.load_dictionary()
    mm = store.load_corpus()

    engine_option = options['engine_option']

    if 'MALLET' in options.get('engine_name', '').upper():

        engine_option.update({ 'prefix': store.target_folder, "workers": 4, "optimize_interval": 10 })
        mallet_home = engine_option.get('engine_path', '')
        model_mallet = LdaMalletService(mm, id2word=dictionary, default_mallet_home=mallet_home, **engine_option)
        FileUtility.compress(model_mallet.ftopicwordweights())
        store.store_mallet_model(model_mallet)

        model_gensim = models.wrappers.ldamallet.malletmodel2ldamodel(model_mallet, iterations=engine_option.get('iterations', 2000))

        result_model = model_mallet

    else:  # if 'MALLET' in options.get('engine_name', '').upper() == 'gensim.models.ldamodel':

        engine_option.update({ 'dtype': np.float64, 'alpha': 'auto' })
        model_gensim = models.LdaModel(corpus=mm, id2word=dictionary, **engine_option)
        result_model = model_gensim

    # Persist gensim model to disk
    store.store_gensim_lda_model(model_gensim)

    return result_model
