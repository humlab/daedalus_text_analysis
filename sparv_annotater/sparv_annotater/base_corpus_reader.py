# -*- coding: utf-8 -*-
import os
from lxml import etree
import zipfile
from io import StringIO
import gensim
#from common.utility import isfileext
import logging

logger = logging.getLogger(__name__)

class BaseCorpusReader():
  
    def __init__(self, source, transforms, chunk_size=None, filetype=''):

        self.filetype = filetype
        self.source = source
        self.chunk_size = chunk_size or 10000
        self.tokenize = gensim.utils.tokenize
        self.transforms = [ self.remove_empty ] + (transforms or [])


    def apply_transforms(self, tokens):
        for ft in self.transforms:
            tokens = ft(tokens)
        return tokens

    def remove_empty(self, t):
        return [ x for x in t if x != '' ]

    def document_iterator(self, content):
        tokens = list(self.tokenize(content))
        tokens = self.apply_transforms(tokens)
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
                        filenames = [ x for x in zf.namelist() if x.endswith(self.filetype) ]
                        for filename in filenames:
                            with zf.open(filename) as text_file:
                                content = text_file.read().decode('utf8')
                            if content == '':
                                continue
                            yield (filename, content)
                elif source.endswith(self.filetype):
                    with open(source, 'r', encoding='utf-8') as f:
                        content = f.read()
                    yield (source, content)
            elif os.path.isdir(source):
                logger.error("Path: source not implemented!")
                raise Exception("Path: source not implemented!")
            else:
                raise Exception("Unable to determine type of source (file not found)")

    def __iter__(self):

        for (filename, content) in self.documents_iterator(self.source):
            basename, _ = os.path.splitext(filename)
            chunk_counter = 0
            for chunk_tokens in self.document_iterator(content):
                chunk_counter += 1
                if len(chunk_tokens) == 0:
                    continue
                yield '{}_{}.txt'.format(basename, str(chunk_counter).zfill(2)), chunk_tokens

class TextCorpusReader(BaseCorpusReader):
  
    def __init__(self, source, transforms, chunk_size=None):
        BaseCorpusReader.__init__(self, source, transforms, chunk_size, filetype='txt')

    