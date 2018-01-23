
import os
import pyLDAvis
import pyLDAvis.gensim

from . import ModelUtility

def extend(a, b):
    return a.update(b) or a

def convert_to_pyLDAvis(data_folder, basename, **opts):

    opts = extend(dict(R=50, mds='tsne', sort_topics=False, plot_opts={'xlab': 'PC1', 'ylab': 'PC2'}), opts or {})

    lda = ModelUtility.load_gensim_lda_model(data_folder, basename)
    corpus = ModelUtility.load_corpus(data_folder, basename)

    data = pyLDAvis.gensim.prepare(lda, corpus, lda.dictionary, **opts)

    # pyLDAvis.save_json(data, os.path.join(target_folder, 'pyldavis.json'))
    pyLDAvis.save_html(data, os.path.join(data_folder, basename, 'pyldavis.html'))

    return data
