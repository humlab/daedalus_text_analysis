import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import os
from topic_modelling import ModelUtility
from topic_modelling import convert_to_pyLDAvis
import logging
from gensim.corpora import MmCorpus, Dictionary
from gensim.models.ldamodel import LdaModel

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

        target_folder = os.path.join(data_folder, basename)
        corpus_filename = os.path.join(target_folder, 'corpus.mm')
        dictionary_filename = os.path.join(target_folder, 'corpus.dict.gz')
        model_filename = os.path.join(target_folder, 'gensim_model_{}.gensim.gz'.format(basename))

        lda = LdaModel.load(model_filename)
        dictionary = Dictionary.load(dictionary_filename)
        corpus = MmCorpus(corpus_filename)

        convert_to_pyLDAvis(data_folder, basename, **OPTS)
