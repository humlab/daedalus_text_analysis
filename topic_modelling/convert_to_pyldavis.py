import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import os
from . compute_topic_models.model_store import ModelStore as store
from . compute_topic_models import convert_to_pyLDAvis

if __name__ == '__main__':

    '''
    What           Visualize LDA topic model using pyLDAvis
    Documentation  https://pyldavis.readthedocs.io/en/latest/
    Source         https://github.com/bmabey/pyLDAvis
    Article        https://nlp.stanford.edu/events/illvi2014/papers/sievert-illvi2014.pdf
    '''
    source_folder = '/tmp/Data'
    models_names = store.get_model_names(source_folder)

    print(models_names)
    OPTS = { 'R': 100, 'mds': 'tsne', 'sort_topics': False, 'plot_opts': { 'xlab': 'PC1', 'ylab': 'PC2' } }

    for basename in models_names:

        data_folder = os.path.join(source_folder, basename)

        lda = store.load_gensim_lda_model(data_folder, basename)
        dictionary = store.load_dictionary(data_folder)
        corpus = store.load_corpus(data_folder)

        convert_to_pyLDAvis(lda, corpus, dictionary, target_folder=data_folder, **OPTS)
        