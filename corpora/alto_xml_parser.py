# -*- coding: utf-8 -*-
import os
import logging
from lxml import etree
from io import StringIO

logger = logging.getLogger(__name__)

script_path = os.path.dirname(os.path.abspath( __file__ ))
XSLT_FILENAME = os.path.join(script_path, 'alto_xml_extract.xslt')

class AltoXmlToText():

    def __init__(self, xslt_filename=None, postags=None, lemmatize=True, deliminator="|"):

        self.xslt_filename = xslt_filename or XSLT_FILENAME
        self.postags = "'{}'".format(postags) if postags is not None else ''
        self.xslt = etree.parse(self.xslt_filename)
        self.xslt_transformer = etree.XSLT(self.xslt)
        self.deliminator = "'{}'".format(deliminator)
        self.lemmatize = lemmatize

    def transform(self, content):
        xml = etree.parse(StringIO(content))
        target = "'lemma'" if self.lemmatize is True else "'content'"
        text = self.xslt_transformer(xml, postags=self.postags, deliminator=self.deliminator, target=target)
        return str(text)
