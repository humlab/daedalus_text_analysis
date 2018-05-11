
import os
import pyLDAvis
import pyLDAvis.gensim

from .model_utility import ModelUtility
from gensim.corpora import MmCorpus
from gensim.models.ldamodel import LdaModel

def extend(a, b):
    return a.update(b) or a

def convert_to_pyLDAvis(data_folder, basename, **opts):

    opts = extend(dict(R=50, mds='tsne', sort_topics=False, plot_opts={'xlab': 'PC1', 'ylab': 'PC2'}), opts or {})

    target_folder = os.path.join(data_folder, basename)

    corpus_filename = os.path.join(target_folder, 'corpus.mm')
    model_filename = os.path.join(target_folder, 'gensim_model_{}.gensim.gz'.format(basename))

    lda = LdaModel.load(model_filename)
    corpus = MmCorpus(corpus_filename)

    data = pyLDAvis.gensim.prepare(lda, corpus, lda.id2word, **opts)

    pyLDAvis.save_html(data, os.path.join(target_folder, 'pyldavis.html'))

    return data
