# -*- coding: utf-8 -*-
import os
from lxml import etree
import zipfile
from io import StringIO
import gensim
from common.utility import isfileext

script_path = os.path.dirname(os.path.abspath( __file__ ))
XSLT_FILENAME = os.path.join(script_path, 'extract_tokens.xslt')

class SparvCorpusReader():
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
    def __init__(self, source, postags=None, chunk_size=None, lowercase=True, min_token_size=3, xslt_filename=None, deliminator="|", lemmatize=True):
        self.xslt_filename = xslt_filename or XSLT_FILENAME
        self.source = source
        self.postags = postags if postags is not None else ''
        self.chunk_size = chunk_size or 10000
        self.lowercase = lowercase
        self.xslt = etree.parse(self.xslt_filename)
        self.transformer = etree.XSLT(self.xslt)
        self.min_token_size = min_token_size
        self.deliminator = deliminator
        self.tokenize = self.sparv_tokenize
        self.lemmatize = lemmatize
        self.transforms = [ self.remove_empty ]

        if self.lowercase is True:
            self.transforms.append(lambda _tokens: list(map(lambda y: y.lower(), _tokens)))

        if self.min_token_size > 0:
            self.transforms.append(lambda _tokens: [ x for x in _tokens if len(x) >= self.min_token_size ])

    def apply_transforms(self, tokens):
        for ft in self.transforms:
            tokens = ft(tokens)
        return tokens

    def remove_empty(self, t):
        return [ x for x in t if x != '' ]

    def sparv_tokenize(self, text, **args):
        return str(text).split(self.deliminator)  # gensim.utils.tokenize

    def document_iterator(self, content):
        xml = etree.parse(StringIO(content))
        target = "'lemma'" if self.lemmatize is True else "'content'"
        text = self.transformer(xml, postags=self.postags, deliminator="'{}'".format(self.deliminator), target=target)
        tokens = list(self.tokenize(text))
        tokens = self.apply_transforms(tokens)
        if self.min_token_size > 1:
            tokens = [ x for x in tokens if len(x) >= self.min_token_size ]
        if self.chunk_size is None:
            yield tokens
        else:
            for i in range(0, len(tokens), self.chunk_size):
                yield tokens[i: i + self.chunk_size]

    def documents_iterator(self, source):
        if isinstance(source, (list,)):
            for document, content in source:
                yield (document, content)
        elif isinstance(source, str):
            if os.path.isfile(source):
                if source.endswith('zip'):
                    with zipfile.ZipFile(source) as zf:
                        filenames = [ x for x in zf.namelist() if x.endswith("xml") ]
                        for filename in filenames:
                            with zf.open(filename) as text_file:
                                content = text_file.read().decode('utf8')
                            if content == '':
                                continue
                            yield (filename, content)
                elif source.endswith('xml'):
                    with open(source, 'r', encoding='utf-8') as f:
                        content = f.read()
                    yield (source, content)
            elif os.path.isdir(source):
                print("Path: source not implemented!")
                raise Exception("Path: source not implemented!")
                
    def __iter__(self):

        for (filename, content) in self.documents_iterator(self.source):
            basename, _ = os.path.splitext(filename)
            chunk_counter = 0
            for chunk_tokens in self.document_iterator(content):
                chunk_counter += 1
                if len(chunk_tokens) == 0:
                    continue
                yield '{}_{}.txt'.format(basename, str(chunk_counter).zfill(2)), chunk_tokens
