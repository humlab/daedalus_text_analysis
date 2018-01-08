import gzip, shutil
import os
from nltk.tokenize import sent_tokenize, word_tokenize
import gensim.models
from gensim.models.word2vec import Word2Vec
from nltk.corpus import stopwords
from w2v_projector import W2V_TensorFlow
import zipfile
import glob
import logging
from common import FileUtility
from . import ModelStore

logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s",  level=logging.INFO)

class CorpusCleanser(object):

    def __init__(self, options):
        self.stopwords = set(stopwords.words('swedish'))
        self.options = options

    def cleanse(self, sentence, min_word_size=2):

        sentence = [ x.lower() for x in sentence ]
        #sentence = [ x for x in sentence if len(x) >= min_word_size ]
        if self.options.get('filter_stopwords', False):
            sentence = [ x for x in sentence if x not in self.stopwords ]
        #sentence = [ x for x in sentence if not x.isdigit() ]
        sentence = [ x for x in sentence if any(map(lambda x: x.isalpha(), x)) ]
        return sentence

    def compress(self, filename):
        with open(filename, 'rb') as f_in:
            with gzip.open(filename + '.gz', 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

class Word2Vectorizer(object):

    def __init__(self, options):
        self.options = options

    def process(self, sentences):

        model = Word2Vec(
            sentences,
            size=self.options['size'],
            window=self.options['window'],
            sg=self.options['sg'],
            iter=self.options['iter'],
            min_count=self.options['min_count'],
            workers=self.options['workers']
        )
        return model

class ZipFileSentenizer(object):

    def __init__(self, pattern, cleanser=None, extensions = [ 'txt' ]):

        self.pattern = pattern
        self.cleanser = cleanser
        self.extensions = extensions

    def __iter__(self):

        for zip_path in glob.glob(self.pattern):
            with zipfile.ZipFile(zip_path) as zip_file:
                filenames = [ name for name in zip_file.namelist() if any(map(name.endswith, self.extensions)) ]
                print(filenames)
                for filename in filenames:
                    with zip_file.open(filename) as text_file:
                        content = text_file.read().decode('utf8').replace('-\r\n','').replace('-\n','')
                        if content == '': continue
                        # fix hyphenations i.e. hypens at end om libe
                        for sentence in sent_tokenize(content, language='swedish'):
                            tokens = word_tokenize(sentence)
                            if not self.cleanser is None:
                                tokens = self.cleanser.cleanse(tokens)
                            if len(tokens) > 0:
                                yield tokens

def compute_word2vec(options):

    if options['skip']:
        return

    basename = ModelStore.create_basename(options)
    directory = os.path.join(options['output_path'], basename + '\\')

    repository = FileUtility(directory)
    repository.create(True)

    sentences = ZipFileSentenizer(options['input_path'], CorpusCleanser(options))

    if options.get('bigram_transformer', False):
        bigram_transformer = gensim.models.phrases.Phraser(sentences)
        sentences_iterator =  bigram_transformer[sentences]
    else:
        sentences_iterator =  sentences

    model = Word2Vectorizer(options).process(sentences_iterator)

    model_filename = os.path.join(directory, 'w2v_model_{}.dat'.format(basename))
    model.save(model_filename)

    model_tsv_filename = os.path.join(directory, 'w2v_vector_{}.tsv'.format(basename))
    model.wv.save_word2vec_format(model_tsv_filename)

    W2V_TensorFlow().convert_file(model_filename, dimension=options['size'])
