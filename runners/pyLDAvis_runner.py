import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import os
from topic_modelling import ModelUtility
from topic_modelling import convert_to_pyLDAvis
import logging

logger = logging.getLogger(__name__)

if __name__ == '__main__':

    '''
    What           Visualize LDA topic model using pyLDAvis
    Documentation  https://pyldavis.readthedocs.io/en/latest/
    Source         https://github.com/bmabey/pyLDAvis
    Article        https://nlp.stanford.edu/events/illvi2014/papers/sievert-illvi2014.pdf
    '''
    data_folder = '/tmp/Data'
    models_names = [ ]  # store.get_model_names(data_folder)

    logger.info(models_names)
    OPTS = { 'R': 100, 'mds': 'tsne', 'sort_topics': False, 'plot_opts': { 'xlab': 'PC1', 'ylab': 'PC2' } }

    for basename in models_names:

        model_folder = os.path.join(data_folder, basename)

        lda = ModelUtility.load_gensim_lda_model(data_folder, basename)
        dictionary = ModelUtility.load_dictionary(data_folder, basename)
        corpus = ModelUtility.load_corpus(data_folder, basename)

        convert_to_pyLDAvis(data_folder, basename, **OPTS)
