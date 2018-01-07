# -*- coding: utf-8 -*-

import logging
import os
import re
from gensim import corpora, models
from gensim.models.wrappers import ldamallet
from gensim.models.ldamodel import LdaModel
from .. sparv_annotator import SparvCorpusReader
import utility
import pandas as pd
import itertools
import inspect
import numpy as np
import pyLDAvis
import pyLDAvis.gensim

default_mallet_path = 'C:\\Usr\\mallet-2.0.8'
logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

class FunctionArgumentUtility:

    @staticmethod
    def filter_by_function(f, args):
        return { k: args[k] for k in args.keys()
            if k in inspect.getfullargspec(f).args }

class MyLdaMallet(models.wrappers.LdaMallet):

    def __init__(self, corpus, id2word, **args):

        args = FunctionArgumentUtility.filter_by_function(super(MyLdaMallet, self).__init__, args)

        args.update({ "workers": 4, "optimize_interval": 10 })

        mallet_home = os.environ.get('MALLET_HOME', default_mallet_path)

        if not mallet_home:
            raise Exception("Environment variable MALLET_HOME not set. Aborting")

        mallet_path = os.path.join(mallet_home, 'bin', 'mallet') if mallet_home else None

        if os.environ.get('MALLET_HOME', '') != mallet_home:
            os.environ["MALLET_HOME"] = mallet_home

        super(MyLdaMallet, self ).__init__(mallet_path, corpus=corpus, id2word=id2word, **args)

    def ftopicwordweights(self):
        return self.prefix + 'topicwordweights.txt'

    def train(self, corpus):
        from gensim.utils import check_output
        self.convert_input(corpus, infer=False)
        cmd = self.mallet_path + ' train-topics --input %s --num-topics %s  --alpha %s --optimize-interval %s '\
            '--num-threads %s --output-state %s --output-doc-topics %s --output-topic-keys %s --topic-word-weights-file %s '\
            '--num-iterations %s --inferencer-filename %s --doc-topics-threshold %s'
        cmd = cmd % (
            self.fcorpusmallet(), self.num_topics, self.alpha, self.optimize_interval,
            self.workers, self.fstate(), self.fdoctopics(), self.ftopickeys(), self.ftopicwordweights(), self.iterations,
            self.finferencer(), self.topic_threshold
        )
        # NOTE "--keep-sequence-bigrams" / "--use-ngrams true" poorer results + runs out of memory
        logger.info("training MALLET LDA with %s", cmd)
        check_output(args=cmd, shell=True)
        self.word_topics = self.load_word_topics()
        self.wordtopics = self.word_topics

class DaedalusTextCorpus(corpora.TextCorpus):

    def __init__(self, archive_name, pos_tags="'|NN|'", chunk_size=1000, lowercase=True, filter_extreme_args=None):

        self.archive_name = archive_name
        self.pos_tags = "'{}'".format(pos_tags)
        self.xslt_filename = './extract_tokens.xslt'
        self.reader = SparvCorpusReader(self.xslt_filename, archive_name, self.pos_tags, chunk_size, lowercase)
        self.document_length = []
        self.corpus_documents = []
        self.filter_extreme_args = filter_extreme_args

        super(DaedalusTextCorpus, self).__init__(input=True) #, token_filters=[])


    def init_dictionary(self, dictionary):
        #self.dictionary = corpora.Dictionary(self.getstream())
        self.dictionary = corpora.Dictionary()
        self.dictionary.add_documents(self.get_texts())
        if self.filter_extreme_args is not None and isinstance(self.filter_extreme_args, dict):
            self.dictionary.filter_extremes(**self.filter_extreme_args)
            self.dictionary.compactify()

    def getstream(self):
        corpus_documents = []
        document_length = []
        for document_name, document in self.reader:
            corpus_documents.append(document_name)
            document_length.append(len(document))
            yield document
        self.document_length = document_length
        self.corpus_documents = corpus_documents

    def get_texts(self):
        for document in self.getstream():
            yield document

    def get_total_word_count(self):
        # Create the defaultdict: total_word_count
        total_word_count = { word_id: 0 for word_id in self.dictionary.keys() }
        for word_id, word_count in itertools.chain.from_iterable(self):
            total_word_count[word_id] += word_count

        # Create a sorted list from the defaultdict: sorted_word_count
        sorted_word_count = sorted(total_word_count, key=lambda w: w[1], reverse=True)
        return sorted_word_count

    def get_corpus_document(self):
        df = pd.DataFrame({'document': self.corpus_documents,'length': self.document_length })
        df['year'] = df.document.apply(lambda x: int(re.search('(\d{4})', x).group(0)))
        return df

def create_base_name(opt):
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
        '_{}'.format(opt.get('lda_engine', '').lower())
    )



class LdaModelHelper():

    @staticmethod
    def convert_to_pyLDAvis(lda, corpus, dictionary, R=50, mds='tsne', sort_topics=False, plot_opts={'xlab': 'PC1', 'ylab': 'PC2'}, target_folder=None):
        data = pyLDAvis.gensim.prepare(lda, corpus, dictionary, R=R, mds=mds, plot_opts=plot_opts, sort_topics=sort_topics)
        if target_folder is not None:
            # pyLDAvis.save_json(data, os.path.join(target_folder, 'pyldavis.json'))
            pyLDAvis.save_html(data, os.path.join(target_folder, 'pyldavis.html'))
        return data

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
        df = df_topic_token_weights.groupby(['topic_id'])['token'].apply(lambda x: ' '.join(x)).reset_index()
        df['top10'] = df['token'].apply(lambda z: '{}'.format('_'.join(map(lambda x: x.title(), z.split(' ')[:10]))))
        return df

    @staticmethod
    def get_yearly_mean_topic_weight(df_doc_topic_weights, df_topic_overview):
        df = df_doc_topic_weights.groupby(['year', 'topic_id'])['weight'].mean().reset_index()
        df = pd.merge(df, df_topic_overview, how='inner', left_on='topic_id', right_on='topic_id')
        return df[['year', 'topic_id', 'weight', 'top10']]

    #df_doc_topics = pd.read_csv(os.path.join(repository.directory, model.fdoctopics()), sep='\t', header=None, index_col=False)
    #df_doc_topics.columns = [ 'document_id', 'label_id'] + topic_columns

    #df_topic_keys = pd.read_csv(os.path.join(repository.directory, model.ftopickeys()), sep='\t', header=None, index_col=False)
    #df_topic_keys.columns = [ 'topic_id', 'not_used', 'tokens' ]

def compute(archive_name, options, default_opt={}):

    for _opt in options:

        opt = dict(default_opt)
        opt.update(_opt)

        if opt["skip"] is True:
            continue

        lda_options = dict(opt['lda_options'])

        basename = create_base_name(opt)
        directory = os.path.join('C:\\tmp\\', basename + '\\')
        repository = utility.UtilityRepository(directory)
        repository.create(True)

        corpus = DaedalusTextCorpus(archive_name, opt.get("pos_tags"), opt.get("chunk_size", None), opt.get("lowercase", True), opt.get("filter_extremes", False))

        ''' Convert Corpus to Matrix Market format and save to disk... '''
        df_corpus_document = corpus.get_corpus_document()
        corpora.MmCorpus.serialize(os.path.join(repository.directory, 'corpus.mm'), corpus)
        corpus.dictionary.save(os.path.join(repository.directory, 'corpus.dict.gz'))
        df_corpus_document.to_csv(os.path.join(repository.directory, 'corpus_documents.csv'), sep='\t')

        ''' Use mm as corpore instead of (slower) DaedalusTextCorpus... '''
        dictionary = corpora.Dictionary.load(os.path.join(repository.directory, 'corpus.dict.gz'))
        mm = corpora.MmCorpus(os.path.join(repository.directory, 'corpus.mm'))

        if 'MALLET' in opt.get('lda_engine', '').upper():

            lda_options.update({ 'prefix': repository.directory, "workers": 4, "optimize_interval": 10 })

            model = MyLdaMallet(mm, id2word=dictionary, **lda_options)

            # model.save(os.path.join(repository.directory, 'mallet_model_{}.gensim.gz'.format(basename)))

            ''' Convert to Gensim LDA model and save to disk... '''
            lda = ldamallet.malletmodel2ldamodel(model)

            ''' Compress files to save space... '''
            repository.zip(model.ftopicwordweights())

        else:

            lda_options.update({ 'dtype': np.float64 })

            lda = LdaModel(corpus=mm, id2word=dictionary, **lda_options)

        lda_filename = os.path.join(repository.directory, 'gensim_model_{}.gensim.gz'.format(basename))
        lda.save(lda_filename)

        ''' re-read from disk to get a clean model '''
        lda = LdaModel.load(lda_filename)

        ''' Prepare various data frames to be saved as sheets in an Excel file... '''
        df_doc_topic_weights = LdaModelHelper.get_document_topics(lda, mm, df_corpus_document, num_words=200, minimum_probability=0)
        df_topic_token_weights = LdaModelHelper.get_topic_token_weight_toplist(lda, num_words=200)
        df_topic_overview = LdaModelHelper.get_topic_overview(df_topic_token_weights)
        df_yearly_mean_topic_weights = LdaModelHelper.get_yearly_mean_topic_weight(df_doc_topic_weights, df_topic_overview)
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
        LdaModelHelper.convert_to_pyLDAvis(lda, mm, dictionary, R=100, mds='tsne', sort_topics=False, target_folder=repository.directory)

