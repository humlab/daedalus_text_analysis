# -*- coding: utf-8 -*-
import os
import re
import gensim
import logging
from . alto_xml_parser import AltoXmlToText

logger = logging.getLogger(__name__)
hyphen_regexp = re.compile(r'\b(\w+)-\s*\r?\n\s*(\w+)\b', re.UNICODE)

def remove_empty(t):
    return [ x for x in t if x != '' ]

def basename(path):
    return os.path.splitext(os.path.basename(path))[0]

def remove_hyphens(text):
    result = re.sub(hyphen_regexp, r"\1\2\n", text)
    return result

class CorpusSourceReader():

    def __init__(self, source=None, transforms=None, chunk_size=None, pattern=None, tokenize=None):

        self.pattern = pattern or ''
        
        if isinstance(source, str):
            if os.path.isfile(source):
                self.source = read_textfile(source)                    
            elif os.path.isdir(source):
                self.source = (self.read_textfile(filename) for filename in glob.glob(os.path.join(folder, pattern)))
            else:
                self.source = (('document', x) for x in [source])
        elif isinstance(source, (list,)):
            self.source = ((x,y) for x, y in source)
        else:
            self.source = source
            
        self.chunk_size = chunk_size
        self.tokenize = tokenize or gensim.utils.tokenize
        self.transforms = [ remove_empty ] + (transforms or [])
        
    def read_textfile(self, filename):
        with open(filename, 'r', encoding='utf-8') as f:
            yield (filename, f.read())

    def apply_transforms(self, tokens):
        for ft in self.transforms:
            tokens = ft(tokens)
        return tokens

    def preprocess(self, content):
        return content
    
    def document_iterator(self, content):
        
        content = self.preprocess(content)
        tokens = self.tokenize(content)
        
        for ft in self.transforms:
            tokens = ft(tokens)
        
        if self.chunk_size is None:
            yield 1, tokens
        else:
            chunk_counter = 0
            for i in range(0, len(tokens), self.chunk_size):
                chunk_counter += 1
                yield chunk_counter, tokens[i: i + self.chunk_size]

    def __iter__(self):
        
        for (filename, content) in self.source:  # self.documents_iterator(self.source):
            for chunk_counter, chunk_tokens in self.document_iterator(content):
                if len(chunk_tokens) == 0:
                    continue
                yield '{}_{}.txt'.format(basename(filename), str(chunk_counter).zfill(2)), chunk_tokens

class SparvCorpusSourceReader(CorpusSourceReader):

    def __init__(self, source, transforms=None, postags=None, lemmatize=True, chunk_size=None, xslt_filename=None, deliminator="|"):

        tokenize = lambda x: str(x).split(deliminator)
        
        super(SparvCorpusSourceReader, self).__init__(source, transforms, chunk_size, pattern='*.xml', tokenize=tokenize)

        self.alto_parser = AltoXmlToText(xslt_filename=xslt_filename, postags=postags, lemmatize=lemmatize)

    def preprocess(self, content):
        return self.alto_parser.transform(content)
    
class TextCorpusSourceReader(CorpusSourceReader):

    def __init__(self, source, transforms, chunk_size=None):
        CorpusSourceReader.__init__(self, source, transforms, chunk_size, pattern='*.txt')
