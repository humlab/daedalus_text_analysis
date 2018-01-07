# -*- coding: utf-8 -*-
import os
from lxml import etree
import zipfile
from io import StringIO
import gensim
import itertools

class SparvCorpusReader():
    '''
    Opens a ZIP-file containing Sparv XML files and extracts lemmatized tokens using an XSLT tranformation.
    Returns a list of tokens filtered by POS in blocks of chunk_size tokens.
    The XSLT-schema accepts a list of POS-tags, as a string "|tag1|tag2|...|tagn|" e.g. "|NN|VB|"
    The first token, delimited by '|', in the "lemma" attribute of a word-tag is selected
       e.g. 'lång' is selected if tag looks like '<w lemma="|lång|långt|" ...>långt</w>'
    If the "lemma" tag is empty ('|') then the text inside the tag is selected
       e.g. 'lång' is selected if tag looks like '<w lemma="|lång|långt|" ...>långt</w>'
    The returned text is lowercased depending of value of lowercase constructor argument.
    The returned text is splitted into text blocks with a size specified by 'chunk_size'.
    If chuck_size is None, then the entire text is returned.
    '''
    def __init__(self, xslt_filename, archive_name, pos_tags, chunk_size=1000, lowercase=True, min_token_size=2):
        self.xslt_filename = xslt_filename
        self.archive_name = archive_name
        self.pos_tags = pos_tags
        self.chunk_size = chunk_size
        self.lowercase = lowercase
        self.xslt = etree.parse(xslt_filename)
        self.transformer = etree.XSLT(self.xslt)
        self.min_token_size=min_token_size

    def document_iterator(self, content):
        xml = etree.parse(StringIO(content))
        text = self.transformer(xml, pos_tags=self.pos_tags)
        tokens = list(gensim.utils.tokenize(text, lowercase=self.lowercase))
        if self.min_token_size > 1:
            tokens = [ x for x in tokens if len(x) >= self.min_token_size ]
        if self.chunk_size is None:
            yield tokens
        else:
            for i in range(0, len(tokens), self.chunk_size):
                yield tokens[i:i+self.chunk_size]

    def documents_iterator(self, archive_name):

        with zipfile.ZipFile(archive_name) as zf:
            filenames = [x for x in zf.namelist() if x.endswith("xml")]
            for filename in filenames:
                with zf.open(filename) as text_file:
                    content = text_file.read().decode('utf8')
                if content == '':
                    continue
                yield (filename, content)

    def __iter__(self):

        for (filename, content) in self.documents_iterator(self.archive_name):
            basename, _ = os.path.splitext(filename)
            chunk_counter = 0
            for chunk_tokens in self.document_iterator(content):
                chunk_counter += 1
                if len(chunk_tokens) == 0:
                    continue
                yield '{}_{}.txt'.format(basename, str(chunk_counter).zfill(2)), chunk_tokens

class SparvTextCorpus(gensim.corpora.TextCorpus):

    def __init__(self, archive_name, pos_tags="'|NN|'", chunk_size=1000, lowercase=True, filter_extreme_args=None):

        self.archive_name = archive_name
        self.pos_tags = "'{}'".format(pos_tags)
        self.xslt_filename = './extract_tokens.xslt'
        self.reader = SparvCorpusReader(self.xslt_filename, archive_name, self.pos_tags, chunk_size, lowercase)
        self.document_length = []
        self.corpus_documents = []
        self.filter_extreme_args = filter_extreme_args

        super(SparvTextCorpus, self).__init__(input=True) #, token_filters=[])

    def init_dictionary(self, dictionary):
        #self.dictionary = corpora.Dictionary(self.getstream())
        self.dictionary = gensim.corpora.Dictionary()
        self.dictionary.add_documents(self.get_texts())
        if self.filter_extreme_args is not None and isinstance(self.filter_extreme_args, dict):
            self.dictionary.filter_extremes(**self.filter_extreme_args)
            self.dictionary.compactify()

    def getstream(self):
        corpus_documents = []
        document_length = []
        for document_name, document in self.reader:
            corpus_documents.append(document_name)
            document_length.append(len(document))
            yield document
        self.document_length = document_length
        self.corpus_documents = corpus_documents

    def get_texts(self):
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

