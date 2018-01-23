# -*- coding: utf-8 -*-
import os
import sys
import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import gensim
from gensim.models.wrappers import ldamallet
from gensim.models.ldamodel import LdaModel
import glob
import logging

__cwd__ = os.path.abspath(__file__) if '__file__' in globals() else os.getcwd()

sys.path.append(__cwd__)

logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s",  level=logging.INFO)
class ModelUtility():

    @staticmethod
    def create_basename(opt):
        extremes = opt.get("filter_extreme_args", {})
        lda_opts = opt.get("lda_options", {})
        return "{}{}{}{}{}{}{}{}{}".format(
            'topics_{}'.format(lda_opts.get("num_topics", 0)),
            '_'.join(opt["pos_tags"].split('|')),
            '_no_chunks' if opt.get("chunk_size", None) is None else 'bz_{}'.format(opt.get("chunk_size", 0)),
            '_iterations_{}'.format(lda_opts.get("iterations", 0)),
            '_lowercase' if opt.get("lowercase", False) else '',
            '_keep_{}'.format(extremes.get('keep_n', 0)) if extremes is not None and extremes.get('keep_n', 0) > 0 else '',
            '_no_below_dfs_{}'.format(extremes.get('no_below', 0)) if extremes is not None and extremes.get('no_below', 0) > 0 else '',
            '_no_above_{}'.format(extremes.get('no_above', 0)) if extremes is not None and extremes.get('no_above', 0) > 0 else '',
            '_{}'.format(opt.get('lda_engine', '').lower()))
                
    # @staticmethod
    # def load_mallet_lda_model(data_folder, basename):
    #     filename = os.path.join(data_folder, basename, 'mallet_model_{}.gensim'.format(basename))
    #     mallet_lda = ldamallet.LdaMallet.load(filename, mmap=None)
    #     lda = ldamallet.malletmodel2ldamodel(mallet_lda)
    #     return lda

    @staticmethod
    def load_gensim_lda_model(data_folder, basename):
        filename = os.path.join(data_folder, basename, 'gensim_model_{}.gensim.gz'.format(basename))
        lda = LdaModel.load(filename)
        return lda

    @staticmethod
    def store_gensim_lda_model(data_folder, basename, lda):
        filename = os.path.join(data_folder, basename, 'gensim_model_{}.gensim.gz'.format(basename))
        lda.save(filename)

    @staticmethod
    def load_dictionary(data_folder, basename):
        filename = os.path.join(data_folder, basename, 'corpus.dict.gz')
        dictionary = gensim.corpora.Dictionary.load(filename)
        return dictionary

    @staticmethod
    def store_dictionary(data_folder, basename, dictionary):
        filename = os.path.join(data_folder, basename, 'corpus.dict.gz')
        dictionary.save(filename)

    @staticmethod
    def store_corpus(data_folder, basename, corpus):
        filename = os.path.join(data_folder, basename, 'corpus.mm')
        gensim.corpora.MmCorpus.serialize(filename, corpus)

    @staticmethod
    def load_corpus(data_folder, basename):
        filename = os.path.join(data_folder, basename, 'corpus.mm')
        corpus = gensim.corpora.MmCorpus(filename)
        return corpus

    @staticmethod
    def store_document_index(data_folder, basename, documents):
        filename = os.path.join(data_folder, basename, 'document_index.csv')
        pd.DataFrame(documents).to_csv(filename)

    @staticmethod
    def load_document_index(data_folder, basename):
        filename = os.path.join(data_folder, basename, 'document_index.csv')
        df = pd.read_csv(filename)
        return df

    @staticmethod
    def load_result_excel_sheet(data_folder, basename, sheet):
        filename = os.path.join(data_folder, basename, 'result_{}.xlsx'.format(basename))
        with pd.ExcelFile(filename) as xls:
            df = pd.read_excel(xls, sheet)
        return df

    @staticmethod
    def load_topic_tokens(data_folder, basename):
        return ModelUtility.load_result_excel_sheet(data_folder, basename, 'topic_token_weights')

    @staticmethod
    def load_mallet_document_topics(data_folder, basename, melt=False):
        filename = os.path.join(data_folder, basename, 'doctopics.txt')
        df = pd.read_table(filename, header=None, index_col=0)
        n_topics = len(df.columns) - 1
        df.columns = ['document_id'] + list(range(0, n_topics))
        if melt:
            df = pd.melt(
                df,
                id_vars=['document_id'],
                var_name="topic_id",
                value_name="weight",
                value_vars=list(range(0, n_topics))
            )
        return df

    @staticmethod
    def load_document_topics(data_folder, basename, melt=False):
        df = ModelUtility.load_result_excel_sheet(data_folder, basename, 'doc_topic_weights')
        n_topics = len(df.columns) - 3
        document_columns = [ 'document_id', 'document', 'year' ]
        topic_columns = list(range(0, n_topics))
        df.columns = document_columns + topic_columns
        if melt:
            df = pd.melt(
                df,
                id_vars=document_columns,
                var_name="topic_id",
                value_name="weight",
                value_vars=topic_columns)
        return df

    @staticmethod
    def get_model_names(source_folder):
        return [ os.path.split(x)[1] for x in glob.glob(os.path.join(source_folder, '*')) ]

