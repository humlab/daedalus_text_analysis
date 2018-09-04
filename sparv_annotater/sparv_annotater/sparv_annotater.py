# -*- coding: utf-8 -*-
import io
import os
import requests
from lxml import etree
import zipfile
import html
import urllib
import json
import pycurl
from io import BytesIO
import logging

logger = logging.getLogger(__name__)

class SparvPoster_pycurl():

    def request(self, url, headers, params, text):
        buffer = BytesIO()
        curl = pycurl.Curl()
        curl.setopt(curl.POST, True)
        curl.setopt(curl.URL, url)
        curl.setopt(curl.WRITEDATA, buffer)
        # curl.setopt(curl.FOLLOWLOCATION, True)
        curl.setopt(pycurl.HTTPHEADER, [x + ':' + y for x, y in headers.items()])
        curl.setopt(curl.HTTPPOST, [
            ('files[]', (
                curl.FORM_BUFFER, 'fileupload.txt',
                curl.FORM_BUFFERPTR, text.encode('utf-8'),
            )),
        ])
        curl.setopt(pycurl.SSL_VERIFYPEER, 0)
        curl.setopt(pycurl.SSL_VERIFYHOST, 0)
        curl.perform()
        curl.close()

        body = buffer.getvalue()
        return body.decode('utf-8')

import re
hyphen_regexp = re.compile(r'\b(\w+)-\s*\r?\n\s*(\w+)\b', re.UNICODE)

class AnnotateService:

    def __init__(self, settings=None, transforms=[]):

        self.url = 'https://ws.spraakbanken.gu.se/ws/sparv/v2//upload?'
        self.headers = {
            'Origin': 'https://spraakbanken.gu.se',
            'Accept-Encoding': 'gzip, deflate, br',
            'Accept-Language': 'en-US,en;q=0.9,sv-SE;q=0.8,sv;q=0.7',
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/62.0.3202.94 Safari/537.36',
            'Accept': '*/*',
            'Connection': 'keep-alive',
            'DNT': '1',
        }

        self.params = [
            ('settings', '/{"corpus":"untitled","lang":"sv","textmode":"file","word_segmenter":"default_tokenizer","sentence_segmentation":/{"sentence_chunk":"paragraph","sentence_segmenter":"default_tokenizer"/},"paragraph_segmentation":/{"paragraph_chunk":"root","paragraph_segmenter":"blanklines"/},"root":/{"tag":"text","attributes":/[/]/},"extra_tags":/[/],"positional_attributes":/{"lexical_attributes":/["pos","msd","lemma","lex","sense"/],"compound_attributes":/["complemgram","compwf"/],"dependency_attributes":/["ref","dephead","deprel"/],"sentiment":/[/]/},"named_entity_recognition":/["ex","type","subtype"/],"text_attributes":/{"readability_metrics":/[/]/}/}'),
        ]

        self.settings = settings if settings is not None else {
            "corpus": "untitled",
            "lang": "sv",
            "textmode": "file",
            "word_segmenter": "default_tokenizer",
            "sentence_segmentation": {
                "sentence_chunk": "paragraph",
                "sentence_segmenter": "default_tokenizer"
            },
            "paragraph_segmentation": {
                "paragraph_chunk": "root",
                "paragraph_segmenter": "blanklines"
            },
            "root": {
                "tag": "text",
                "attributes": []
            },
            "extra_tags": [],
            "positional_attributes": {
                "lexical_attributes": ["pos", "msd", "lemma", "lex", "sense"],
                "compound_attributes": ["complemgram", "compwf"],
                "dependency_attributes": ["ref", "dephead", "deprel"],
                "sentiment": []
            },
            "named_entity_recognition": ["ex", "type", "subtype"],
            "text_attributes": {
                "readability_metrics": []
            }
        }

        self.transforms = transforms or []
        self.transforms.append(self.remove_hyphens)
        self.transforms.append(html.escape)

    def remove_hyphens(self, text):
        result = re.sub(hyphen_regexp, r"\1\2\n", text)
        return result

    def apply_transforms(self, text):
        for ft in self.transforms:
            text = ft(text)
        return text

    def annotate_text_file(self, source_filename, target_filename):
        with io.open(source_filename, 'r', encoding='utf8') as f:
            text = f.read()
        return self.annotate_text(text, target_filename)

    def annotate_text(self, text, target_filename):

        data = self.apply_transforms(text)
        settings = urllib.parse.quote(json.dumps(self.settings).replace(" ", ""), safe='/{}[]:,')
        url = self.url + "settings=" + settings

        response_text = SparvPoster_pycurl().request(url, self.headers, None, data)

        if response_text is None:
            return None

        url = self.parse_response(response_text)
        if url == '':
            return None

        return self.download_file(url, target_filename)

    def parse_response(self, response_text):
        root = etree.fromstring(response_text)
        if len(root.xpath("//corpus")) > 0:
            return root.xpath("//corpus")[0].get('link', '')
        return ''

    def download_file(self, url, local_filename):
        r = requests.get(url, stream=True)
        with open(local_filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
        return local_filename


class ArchiveAnnotater:

    def __init__(self, settings=None, transforms=[]):
        self.service = AnnotateService(settings, transforms)

    def annotate_content(self, content, target_file):

        self.service.annotate_text(content, target_file)

        with zipfile.ZipFile(target_file) as zf:
            result_name = next(iter([x for x in zf.namelist() if x.endswith('xml')]))

            with zf.open(result_name) as pos_file:
                result_content = pos_file.read().decode('utf8')
                return result_content

    def annotate_files_in_archive(self, source_filename, target_filename, write_xml_file=False, delete_sparv_file=True):

        target_folder, _ = os.path.split(target_filename)

        result_zf = zipfile.ZipFile(target_filename, 'w', zipfile.ZIP_DEFLATED)
        with zipfile.ZipFile(source_filename) as zf:
            namelist = [x for x in zf.namelist() if x.endswith('txt')]
            file_count = len(namelist)
            counter = 0
            for article_name in namelist:

                basename = os.path.splitext(article_name)[0]
                download_name = os.path.join(target_folder, basename + '.zip')
                xml_name = basename + '.xml'

                if os.path.isfile(os.path.join(target_folder, xml_name)):
                    logger.warning('WARNING: File {} exists, skipping...'.format(xml_name))
                    continue

                with zf.open(article_name) as tf:
                    content = tf.read().decode('utf8')

                xml = self.annotate_content(content, download_name)

                if xml is None:
                    logger.error('FAILED: {}...'.format(article_name))
                    continue

                result_zf.writestr(xml_name, xml)

                if write_xml_file:
                    with io.open(os.path.join(target_folder, xml_name), 'w', encoding='utf8') as f:
                        f.write(xml)

                if delete_sparv_file:
                    os.remove(download_name)

                counter += 1
                logger.info('DONE: {} ({}) {}...'.format(counter, file_count, article_name))
                # if counter == 1:
                #    break
