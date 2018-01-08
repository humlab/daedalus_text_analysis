# -*- coding: utf-8 -*-

import logging
import os
import re
from gensim import corpora, models, matutils
from . import SparvTextCorpus
import pandas as pd
import numpy as np
from model_store import ModelStore as store
from sklearn.feature_extraction.text import TfidfVectorizer
from . import convert_to_pyLDAvis
from . import LdaMalletService

class ModelComputeHelper():

    @staticmethod
    def get_topic_token_weight_toplist(lda, num_words=200):
        df_topic_weights = pd.DataFrame(
            [ (topic_id, token, weight)
                for topic_id, tokens in (lda.show_topics(lda.num_topics, num_words=num_words, formatted=False))
                    for token, weight in tokens if weight > 0.0 ],
            columns=['topic_id', 'token', 'weight']
        )
        return df_topic_weights

    @staticmethod
    def get_document_topics(lda, mm, df_corpus_document, num_words=200, minimum_probability=None):

        ''' Get document topic weights for all documents in corpus '''
        ''' Note!  minimum_probability=None filters less probable topics, set to 0 to retrieve all topcs'''
        df_doc_topics = pd.DataFrame(sum([ [ (i, x[0], x[1]) for x in topics ]
            for i, topics in enumerate(lda.get_document_topics(mm, minimum_probability=minimum_probability)) ], []),
                columns = ['document_id', 'topic_id', 'weight'])

        df = pd.merge(df_corpus_document, df_doc_topics, how='inner', left_index=True, right_on='document_id')
        return df

    @staticmethod
    def get_topic_overview(df_topic_token_weights):
        df = df_topic_token_weights.groupby(['topic_id'])['token'].apply(' '.join).reset_index()
        df['top10'] = df['token'].apply(lambda z: '{}'.format('_'.join(map(lambda x: x.title(), z.split(' ')[:10]))))
        return df

    @staticmethod
    def get_yearly_mean_topic_weight(df_doc_topic_weights, df_topic_overview):
        df = df_doc_topic_weights.groupby(['year', 'topic_id'])['weight'].mean().reset_index()
        df = pd.merge(df, df_topic_overview, how='inner', left_on='topic_id', right_on='topic_id')
        return df[['year', 'topic_id', 'weight', 'top10']]

    @staticmethod
    def get_corpus_document(sparvCorpus):
        df = pd.DataFrame({'document': sparvCorpus.corpus_documents,'length': sparvCorpus.document_length })
        df['year'] = df.document.apply(lambda x: int(re.search('(\d{4})', x).group(0)))
        return df

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
        dictionary = Dictionary.from_corpus(corpus_vect_gensim,
                id2word=dict((id, word) for word, id in corpus_vect.vocabulary_.items()))
        return corpus_vect_gensim, dictionary

def compute(archive_name, options, default_opt={}, target_folder='/tmp/'):

    for _opt in options:

        opt = dict(default_opt)
        opt.update(_opt)

        if opt["skip"] is True:
            continue

        lda_options = dict(opt['lda_options'])

        basename = store.create_base_name(opt)
        directory = os.path.join(target_folder, basename + '\\')
        repository = store.UtilityRepository(directory)
        repository.create(True)

        corpus = SparvTextCorpus(archive_name, opt.get("pos_tags"), opt.get("chunk_size", None), opt.get("lowercase", True), opt.get("filter_extremes", False))

        ''' Convert Corpus to Matrix Market format and save to disk... '''
        df_corpus_document = ModelComputeHelper.get_corpus_document(corpus)
        corpora.MmCorpus.serialize(os.path.join(repository.directory, 'corpus.mm'), corpus)
        corpus.dictionary.save(os.path.join(repository.directory, 'corpus.dict.gz'))
        df_corpus_document.to_csv(os.path.join(repository.directory, 'corpus_documents.csv'), sep='\t')

        ''' Use mm as corpore instead of (slower) DaedalusTextCorpus... '''
        dictionary = corpora.Dictionary.load(os.path.join(repository.directory, 'corpus.dict.gz'))
        mm = corpora.MmCorpus(os.path.join(repository.directory, 'corpus.mm'))

        if 'MALLET' in opt.get('lda_engine', '').upper():

            lda_options.update({ 'prefix': repository.directory, "workers": 4, "optimize_interval": 10 })
            mallet_path = opt.get('engine_path', '')
            model = LdaMalletService(mm, id2word=dictionary, mallet_path=mallet_path, **lda_options)

            # model.save(os.path.join(repository.directory, 'mallet_model_{}.gensim.gz'.format(basename)))

            ''' Convert to Gensim LDA model and save to disk... '''
            lda = models.ldamallet.malletmodel2ldamodel(model)

            ''' Compress files to save space... '''
            repository.zip(model.ftopicwordweights())

        else:

            lda_options.update({ 'dtype': np.float64 })

            lda = models.LdaModel(corpus=mm, id2word=dictionary, **lda_options)

        lda_filename = os.path.join(repository.directory, 'gensim_model_{}.gensim.gz'.format(basename))
        lda.save(lda_filename)

        ''' re-read from disk to get a clean model '''
        lda = store.load(lda_filename)

        ''' Prepare various data frames to be saved as sheets in an Excel file... '''
        df_doc_topic_weights = ModelComputeHelper.get_document_topics(lda, mm, df_corpus_document, num_words=200, minimum_probability=0)
        df_topic_token_weights = ModelComputeHelper.get_topic_token_weight_toplist(lda, num_words=200)
        df_topic_overview = ModelComputeHelper.get_topic_overview(df_topic_token_weights)
        df_yearly_mean_topic_weights = ModelComputeHelper.get_yearly_mean_topic_weight(df_doc_topic_weights, df_topic_overview)
        df_dictionary = pd.DataFrame({ 'token': dictionary.id2token, 'dfs': dictionary.dfs }).reset_index().set_index('index')[['token', 'dfs']]

        repository.save_excel(
            [(df_doc_topic_weights.reset_index(), 'doc_topic_weights'),
             (df_yearly_mean_topic_weights, 'yearly_mean_topic_weights'),
             (df_topic_overview, 'topic_tokens'),
             (df_topic_token_weights, 'topic_token_weights'),
             (df_corpus_document, 'documents'),
             (df_dictionary, 'dictionary')
             ],
            os.path.join(repository.directory, 'result_' + basename + '.xlsx')
        )

        ''' Create a pyLDAvis HTML file... '''
        convert_to_pyLDAvis(lda, mm, dictionary, R=100, mds='tsne', sort_topics=False, target_folder=repository.directory)

