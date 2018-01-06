import os
import zipfile
import glob


# class CorpusCleanser(object):
# import shutil
# from nltk.corpus import stopwords
# import gensim.models

#     def __init__(self, options, language='swedish'):
#         self.stopwords = set(stopwords.words(language))
#         self.options = options

#     def cleanse(self, sentence, min_word_size=2):

#         sentence = [x.lower() for x in sentence]
#         # sentence = [ x for x in sentence if len(x) >= min_word_size ]
#         if options.get('filter_stopwords', False):
#             sentence = [x for x in sentence if x not in self.stopwords]
#         # sentence = [ x for x in sentence if not x.isdigit() ]
#         sentence = [x for x in sentence if any(map(lambda x: x.isalpha(), x))]
#         return sentence

#     def compress(self, filename):
#         with open(filename, 'rb') as f_in:
#             with gzip.open(filename + '.gz', 'wb') as f_out:
#                 shutil.copyfileobj(f_in, f_out)


class ZipFileReader(object):

    def __init__(self, pattern, cleanser=None, extensions=['txt']):

        self.pattern = pattern
        self.cleanser = cleanser
        self.extensions = extensions

    def __iter__(self):

        for zip_path in glob.glob(self.pattern):
            with zipfile.ZipFile(zip_path) as zip_file:
                filenames = [name for name in zip_file.namelist() if any(map(lambda x:  name.endswith(x), self.extensions))]
                for filename in filenames:
                    with zip_file.open(filename) as text_file:
                        content = text_file.read().decode('utf8')
                        # content = content.replace('-\r\n','')
                        # content = content.replace('-\n','')
                        if content == '':
                            continue
                        yield (filename, content)


class SegmentSplitter(object):

    def __init__(self, reader, delimiter):
        self.reader = reader
        self.delimiter = delimiter

    def __iter__(self):
        for (filename, content) in self.reader:
            basename, _ = os.path.splitext(filename)
            part_counter = 0
            for article in content.split(self.delimiter):
                article = article.strip()
                if len(article) == 0:
                    continue
                part_counter += 1
                partname = '{}_article_{}.txt'.format(basename, str(part_counter).zfill(2))
                yield (partname, article)


def main():

    path = './data/segmented-yearly-volumes.zip'

    folder, filename = os.path.split(path)
    basename, _ = os.path.splitext(filename)

    splitter = SegmentSplitter(ZipFileReader(path), '###')

    zip_name = os.path.join(folder, '{}_articles_new.zip'.format(basename))

    zf = zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED)

    for (partname, article) in splitter:
        zf.writestr(partname, article)

    zf.close()

if __name__ == "__main__":

    main()
