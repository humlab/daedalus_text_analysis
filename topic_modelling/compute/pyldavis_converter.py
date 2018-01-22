
import os
import pyLDAvis
import pyLDAvis.gensim

def convert_to_pyLDAvis(lda, corpus, dictionary, R=50, mds='tsne', sort_topics=False, plot_opts={'xlab': 'PC1', 'ylab': 'PC2'}, target_folder=None):
    data = pyLDAvis.gensim.prepare(lda, corpus, dictionary, R=R, mds=mds, plot_opts=plot_opts, sort_topics=sort_topics)
    if target_folder is not None:
        # pyLDAvis.save_json(data, os.path.join(target_folder, 'pyldavis.json'))
        pyLDAvis.save_html(data, os.path.join(target_folder, 'pyldavis.html'))
    return data
