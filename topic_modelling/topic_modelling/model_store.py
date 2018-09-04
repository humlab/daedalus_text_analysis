# -*- coding: utf-8 -*-
import os
import sys
import time
import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import gensim
from gensim.models.ldamodel import LdaModel
import glob
import logging

join = os.path.join

__cwd__ = os.path.abspath(__file__) if '__file__' in globals() else os.getcwd()

sys.path.append(__cwd__)

logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelStore():

    def __init__(self, opt):
        self.options = opt
        self.root_folder = opt['root_folder']
        self.basename = self._get_basename()
        self.model_folder = join(self.root_folder, self.basename + '\\')
        self.target_folder = self.model_folder
        self.document_index_filename = join(self.model_folder, 'document_index.csv')
        self.result_excel_filename = join(self.model_folder, 'result_{}.xlsx'.format(self.basename))
        self.mallet_document_topics_filename = join(self.model_folder, 'doctopics.txt')
        self.corpus_mm_filename = join(self.model_folder, 'corpus.mm')
        self.corpus_dict_filename = join(self.model_folder, 'corpus.dict.gz')
        self.gensim_model_filename = join(self.model_folder, 'gensim_model_{}.gensim.gz'.format(self.basename))
        self.mallet_model_filename = join(self.model_folder, 'mallet_model_{}.gensim'.format(self.basename))
        self.mallet_filename = join(self.model_folder, 'mallet.model.gz')

    def _get_basename(self):
        prune_at = self.options.get("prune_at", 2000000)
        dfs_min = self.options.get("dfs_min", 0)
        dfs_max = self.options.get("dfs_max", 0)
        engine_opts = self.options.get("engine_option", {})
        postags = self.options.get("postags", '') or ''
        prefix = self.options.get("prefix", '') or ''
        return "{}{}{}{}{}{}{}{}{}{}".format(
            '{}_{}_'.format(time.strftime("%Y%m%d"), prefix),
            'topics_{}'.format(engine_opts.get("num_topics", 0)),
            '_'.join(postags.split('|')),
            '_no_chunks' if self.options.get("chunk_size", None) is None else '_chunks_{}'.format(self.options.get("chunk_size", 0)),
            '_iterations_{}'.format(engine_opts.get("iterations", 0)),
            '_lowercase' if self.options.get("lowercase", False) else '',
            '_prune_at_{}'.format(prune_at) if prune_at != 2000000 else '',
            '_dfs_min_{}'.format(dfs_min) if dfs_min > 0 else '',
            '_dfs_max_{}'.format(dfs_max) if dfs_max > 0 else '',
            '_{}'.format(self.options.get('engine_name', '').lower()))

    def store_mallet_model(self, model):
        model.save(self.mallet_filename)

    def load_mallet_lda_model(self):
        mallet_lda = gensim.models.ldamallet.LdaMallet.load(self.mallet_model_filename, mmap=None)
        lda = gensim.models.ldamallet.malletmodel2ldamodel(mallet_lda)
        return lda

    def load_gensim_lda_model(self):
        lda = LdaModel.load(self.gensim_model_filename)
        return lda

    def store_gensim_lda_model(self, lda):
        lda.save(self.gensim_model_filename)

    def load_dictionary(self):
        dictionary = gensim.corpora.Dictionary.load(self.corpus_dict_filename)
        return dictionary

    def store_dictionary(self, dictionary):
        dictionary.save(self.corpus_dict_filename)

    def store_corpus(self, corpus):
        gensim.corpora.MmCorpus.serialize(self.corpus_mm_filename, corpus)

    def load_corpus(self):
        corpus = gensim.corpora.MmCorpus(self.corpus_mm_filename)
        return corpus

    def store_document_index(self, documents):
        df = pd.DataFrame(documents).set_index('document_id')
        df.to_csv(self.document_index_filename, sep='\t', header=True)
        return df

    def load_document_index(self):
        df = pd.read_csv(self.document_index_filename, sep='\t', header=0).set_index('document_id')
        return df

    def load_result_excel_sheet(self, sheet):
        with pd.ExcelFile(self.result_excel_filename) as xls:
            df = pd.read_excel(xls, sheet)
        return df

    def load_mallet_topic_keys(self):
        filename = join(self.target_folder, 'topickeys.txt')
        if os.path.isfile(filename):
            df = pd.read_csv(filename, sep='\t', header=None, names=['topic_id', 'alpha', 'tokens'], decimal=b',')
            return df
        return None

    def load_topic_tokens(self):
        return self.load_result_excel_sheet('topic_token_weights')

    def load_mallet_document_topics(self, melt=False):
        df = pd.read_table(self.mallet_document_topics_filename, header=None, index_col=0)
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

    def load_document_topics(self, melt=False):
        df = self.load_result_excel_sheet('doc_topic_weights')
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
        return [ os.path.split(x)[1] for x in glob.glob(join(source_folder, '*')) ]
