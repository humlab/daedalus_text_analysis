import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import os
from .viz_utility import load_gensim_lda_model, load_corpus, load_dictionary
from .viz_utility import get_model_names, convert_to_pyLDAvis

if __name__ == '__main__':

    '''
    What           Visualize LDA topic model using pyLDAvis
    Documentation  https://pyldavis.readthedocs.io/en/latest/
    Source         https://github.com/bmabey/pyLDAvis
    Article        https://nlp.stanford.edu/events/illvi2014/papers/sievert-illvi2014.pdf
    '''
    source_folder = '/tmp/Data'
    models_names = get_model_names(source_folder)

    print(models_names)
    OPTS = { 'R': 100, 'mds': 'tsne', 'sort_topics': False, 'plot_opts': { 'xlab': 'PC1', 'ylab': 'PC2' } }

    for basename in models_names:

        data_folder = os.path.join(source_folder, basename)

        lda = load_gensim_lda_model(data_folder, basename)
        dictionary = load_dictionary(data_folder)
        corpus = load_corpus(data_folder)

        convert_to_pyLDAvis(lda, corpus, dictionary, target_folder=data_folder, basename=basename, **OPTS)

