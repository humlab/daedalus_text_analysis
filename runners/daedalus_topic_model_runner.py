
import re
import sys
import os

__cwd__ = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
__root_path__ = os.path.abspath(os.path.join(__cwd__, '..'))

sys.path.append(__root_path__)

import topic_modelling
import pandas as pd
import zipfile
from sparv_annotater import SparvTextCorpus, SparvCorpusReader
from common.utility import extend, FileUtility

mallet_path = 'C:\\Usr\\mallet-2.0.8'

class DaedalusTopicModelRunner:

    def save_documents(self, corpus, target_folder):
        result_zip_name = os.path.join(target_folder, 'documents.zip')
        with zipfile.ZipFile(result_zip_name, 'w', zipfile.ZIP_DEFLATED) as zf:
            for i, document in enumerate(corpus):
                document_name = '{}.csv'.format(corpus.corpus_documents[i].split('.')[0])
                token_ids, token_counts = list(zip(*document))
                df = pd.DataFrame(dict(token_id=token_ids, token_count=token_counts))
                df['token'] = df.token_id.apply(lambda x: corpus.dictionary[x])
                df.set_index('token_id', inplace=True)
                content = u'\ufeff' + df.to_csv(path_or_buf=None, sep=';', header=True, encoding='utf-8')
                zf.writestr(document_name, content, compress_type=zipfile.ZIP_DEFLATED)

    def create_corpus(self, opt):

        stream = SparvCorpusReader(
            source=opt.get('source', []),
            postags="'{}'".format(opt.get("postags")),
            chunk_size=opt.get("chunk_size", None),
            lowercase=opt.get("lowercase", True),
            min_token_size=opt.get("min_token_size", 3),
            lemmatize=opt.get("lemmatize", True)
        )

        corpus = SparvTextCorpus(stream, prune_at=opt.get("prune_at", 2000000))

        return corpus

    def compute(self, corpus, options):

        return topic_modelling.compute(corpus, options=options)

DEFAULT_OPT = {
    "skip": False,
    "postags": '|NN|PM|',
    "chunk_size": None,
    "lowercase": True,
    "min_token_size": 3,
    "lemmatize": True,
    'lda_engine': 'LdaMallet',
    "lda_options": {
        "num_topics": 50,
        "iterations": 2000,
    },
    'prune_at': 2000000,
    'target_folder': 'C:\\tmp\\',
    'doc_name_attrib_extractors': [('year', lambda x: int(re.search(r'(\d{4})', x).group(0)))]
}


if __name__ == "__main__":

    source = 'C:\\Users\\roma0050\\Documents\\Projects\\daedalus_text_analysis\\data\\daedalus_articles_pos_xml.zip'

    '''
    See https://spraakbanken.gu.se/korp/markup/msdtags.html for description of MSD-tag set
    '''

    for n_topics in [10, 50, 100, 150, 200]:

        options_list = [

            { 'source': source, 'lda_engine': 'LdaMallet', 'lda_options': { "num_topics": n_topics, "iterations": 2000 }, 'engine_path': mallet_path  },

            # { 'source': source, 'lda_engine': 'LdaModel', 'lda_options': { "num_topics": n_topics, "iterations": 2000, 'chunksize': 100000, 'passes': 2  } },
        ]

        for _options in options_list:

            opt = extend(dict(DEFAULT_OPT), _options)

            if opt.get('skip', False) is True:
                continue

            data_folder = opt['target_folder']
            basename = topic_modelling.ModelUtility.create_basename(opt)
            directory = os.path.join(data_folder, basename)

            FileUtility(directory).create(True)

            runner = DaedalusTopicModelRunner()

            corpus = runner.create_corpus(opt)

            if False:
                runner.save_documents(corpus, target_folder=directory)

            model = runner.compute(corpus, opt)

            topic_modelling.generate_notebook_friendly_data(model, data_folder, basename)

            # topic_modelling.convert_to_pyLDAvis(data_folder, basename)
