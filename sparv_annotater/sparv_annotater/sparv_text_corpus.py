import gensim
import itertools

class SparvTextCorpus(gensim.corpora.TextCorpus):

    def __init__(self, stream, filter_extreme_args=None):

        self.dictionary = None
        self.reader = stream
        self.document_length = []
        self.corpus_documents = []
        self.filter_extreme_args = filter_extreme_args

        super(SparvTextCorpus, self).__init__(input=True)  # , token_filters=[])

    def init_dictionary(self, dictionary):
        # self.dictionary = corpora.Dictionary(self.getstream())
        self.dictionary = gensim.corpora.Dictionary()
        self.dictionary.add_documents(self.get_texts())
        if self.filter_extreme_args is not None and isinstance(self.filter_extreme_args, dict):
            self.dictionary.filter_extremes(**self.filter_extreme_args)
            self.dictionary.compactify()

    def getstream(self):
        '''
        Returns stream of documents.
        Also collects documents' name and length for each pass
        '''
        corpus_documents = []
        document_length = []
        for document_name, document in self.reader:
            corpus_documents.append(document_name)
            document_length.append(len(document))
            yield document
        self.document_length = document_length
        self.corpus_documents = corpus_documents

    def get_texts(self):
        '''
        This is mandatory method from gensim.corpora.TextCorpus. Returns stream of documents.
        '''
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
