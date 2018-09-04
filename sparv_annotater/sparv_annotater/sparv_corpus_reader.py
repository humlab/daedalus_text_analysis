# -*- coding: utf-8 -*-
import os
from lxml import etree
import zipfile
from io import StringIO
import gensim
#from common.utility import isfileext
import logging
from . base_corpus_reader import BaseCorpusReader

logger = logging.getLogger(__name__)

script_path = os.path.dirname(os.path.abspath( __file__ ))
XSLT_FILENAME = os.path.join(script_path, 'extract_tokens.xslt')

class SparvCorpusReader(BaseCorpusReader):
    '''
    Opens a ZIP-file containing Sparv XML files and extracts lemmatized tokens using an XSLT tranformation.
    Returns a list of tokens filtered by POS in blocks of chunk_size tokens.
    The XSLT-schema accepts a list of POS-tags, as a string "|tag1|tag2|...|tagn|" e.g. "|NN|VB|"
    The first token, delimited by '|', in the "lemma" attribute of a word-tag is selected
       e.g. 'lång' is selected if tag looks like '<w lemma="|lång|långt|" ...>långt</w>'
    If the "lemma" tag is empty ('|') then the text inside the tag is selected
       e.g. 'långt' is selected if tag looks like '<w lemma="|" ...>långt</w>'
    The returned text is lowercased depending of value of lowercase constructor argument.
    The returned text is splitted into text blocks with a size specified by 'chunk_size'.
    If chuck_size is None, then the entire text is returned.

    Compare to gensim textcorpus rules:
    lowercase and convert to unicode; assumes utf8 encoding
    *deaccent (asciifolding)
    *collapse multiple whitespaces into a single one
    tokenize by splitting on whitespace
    remove words less than 3 characters long
    remove stopwords; see gensim.parsing.preprocessing for the list of stopwords

    source:
        1) ZIP-archive or
        2) list of (document,XML) tuples or
        3) folder with XML files (not implemented)
        4) a single XML file

    Sparv tags:
    https://spraakbanken.gu.se/swe/forskning/infrastruktur/sparv/annotationer

    '''
    def __init__(self, source, transforms, postags=None, chunk_size=None, xslt_filename=None, deliminator="|", lemmatize=True):

        super(SparvCorpusReader, self).__init__(source, transforms, chunk_size, filetype='xml')

        self.xslt_filename = xslt_filename or XSLT_FILENAME
        self.postags = postags if postags is not None else ''
        self.xslt = etree.parse(self.xslt_filename)
        self.xslt_transformer = etree.XSLT(self.xslt)
        self.deliminator = deliminator
        self.tokenize = self.sparv_tokenize
        self.lemmatize = lemmatize
    
    def sparv_tokenize(self, text, **args):
        return str(text).split(self.deliminator)  # gensim.utils.tokenize

    def document_iterator(self, content):
        xml = etree.parse(StringIO(content))
        target = "'lemma'" if self.lemmatize is True else "'content'"
        text = self.xslt_transformer(xml, postags=self.postags, deliminator="'{}'".format(self.deliminator), target=target)
        tokens = list(self.tokenize(text))
        tokens = self.apply_transforms(tokens)
        if self.chunk_size is None:
            yield tokens
        else:
            for i in range(0, len(tokens), self.chunk_size):
                yield tokens[i: i + self.chunk_size]

    