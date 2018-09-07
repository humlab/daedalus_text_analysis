# -*- coding: utf-8 -*-
import os
import zipfile
import glob
import logging

logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO)

class ZipFileIterator(object):

    def __init__(self, pattern, extensions):
        self.pattern = pattern
        self.extensions = extensions

    def __iter__(self):

        for zip_path in glob.glob(self.pattern):
            with zipfile.ZipFile(zip_path) as zip_file:
                filenames = [ name for name in zip_file.namelist() if any(map(name.endswith, self.extensions)) ]
                for filename in filenames:
                    with zip_file.open(filename) as text_file:
                        content = text_file.read().decode('utf8')\
                            .replace('-\r\n', '').replace('-\n', '')
                        yield os.path.basename(filename), content
