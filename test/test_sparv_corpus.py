#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import logging
import unittest
from corpora import SparvCorpusSourceReader, SparvTextCorpus, ZipReader
from sparv_annotater import AnnotateService
from utility import join_test_data_path, generate_temp_filename
from common.utility import extend
from gensim import corpora

import corpora.corpus_source_reader as csr

logger = logging.getLogger(__name__)

def filter_by_token_length(n):
    return (lambda tokens: [ x for x in tokens if len(x) >= n])

def to_lower(tokens):
    return [ x.lower() for x in tokens ]

class SparvCorpusTestCase(unittest.TestCase):

    def setUp(self):

        self.xml_data = '''
            <corpus>
            <paragraph>
            <sentence id="e94c6-e6118">
            <w pos="NN" msd="NN.NEU.SIN.DEF.NOM" lemma="|föremål|" lex="|föremål..nn.1|" sense="|föremål..1:0.816|föremål..2:0.184|" ref="08" dephead="06" deprel="PA">Föremålet</w>
            <w pos="MID" msd="MID" lemma="|" lex="|" sense="|" ref="09" dephead="05" deprel="IK">,</w>
            <w pos="HP" msd="HP.-.-.-" lemma="|" lex="|" sense="|" ref="10" dephead="15" deprel="SS">som</w>
            <w pos="PP" msd="PP" lemma="|förutom|" lex="|förutom..pp.1|" sense="|förutom..1:-1.000|" ref="11" dephead="10" deprel="UA">förutom</w>
            <w pos="JJ" msd="JJ.POS.UTR+NEU.SIN.DEF.NOM" lemma="|själv|själva|" lex="|själv..pn.1|själva..pn.1|" sense="|själv..1:-1.000|själv..2:-1.000|själv..3:-1.000|" ref="12" dephead="13" deprel="DT">själva</w>
            <w pos="NN" msd="NN.UTR.SIN.DEF.NOM" lemma="|" lex="|" sense="|" ref="13" dephead="11" deprel="PA">vertikalfräsmaskinen</w>
            <w pos="MID" msd="MID" lemma="|" lex="|" sense="|" ref="14" dephead="15" deprel="IK">,</w>
            <w pos="VB" msd="VB.PRS.AKT" lemma="|bestå|" lex="|bestå..vb.1|bestå..vb.2|" sense="|bestå..4:0.561|bestå..3:0.221|bestå..1:0.129|bestå..2:0.089|" ref="15" deprel="ROOT">består</w>
            <w pos="PP" msd="PP" lemma="|av|" lex="|av..pp.1|" sense="|av..1:-1.000|" ref="16" dephead="15" deprel="OA">av</w>
            <w pos="DT" msd="DT.NEU.SIN.IND" lemma="|en|" lex="|en..al.1|" sense="|den..1:-1.000|en..2:-1.000|" ref="17" dephead="18" deprel="DT">ett</w>
            <w pos="NN" msd="NN.NEU.SIN.IND.NOM" lemma="|" lex="|" sense="|" ref="18" dephead="19" deprel="CJ">styrskåp</w>
            <w pos="KN" msd="KN" lemma="|och|" lex="|och..kn.1|" sense="|och..1:-1.000|" ref="19" dephead="16" deprel="PA">och</w>
            <w pos="DT" msd="DT.UTR.SIN.IND" lemma="|en|" lex="|en..al.1|" sense="|den..1:-1.000|en..2:-1.000|" ref="20" dephead="21" deprel="DT">en</w>
            <w pos="NN" msd="NN.UTR.SIN.IND.NOM" lemma="|omformare|" lex="|omformare..nn.1|" sense="|omformare..1:-1.000|" ref="21" dephead="19" deprel="CJ">omformare</w>
            <w pos="VB" msd="VB.PRS.AKT" lemma="|utgöra|" lex="|utgöra..vb.1|" sense="|utgöra..1:-1.000|" ref="22" dephead="15" deprel="OO">utgör</w>
            <w pos="DT" msd="DT.NEU.SIN.IND" lemma="|en|" lex="|en..al.1|" sense="|den..1:-1.000|en..2:-1.000|" ref="23" dephead="25" deprel="DT">ett</w>
            <w pos="JJ" msd="JJ.POS.NEU.SIN.IND.NOM" lemma="|viktig|" lex="|viktig..av.1|" sense="|viktig..1:1.000|viktig..2:0.000|" ref="24" dephead="25" deprel="AT">viktigt</w>
            <w pos="NN" msd="NN.NEU.SIN.IND.NOM" lemma="|steg|" lex="|steg..nn.1|" sense="|steg..2:0.940|steg..1:0.060|" ref="25" dephead="22" deprel="OO">steg</w>
            <w pos="PP" msd="PP" lemma="|i|" lex="|i..pp.1|" sense="|i..2:-1.000|" ref="26" dephead="25" deprel="ET">i</w>
            <w pos="NN" msd="NN.UTR.SIN.DEF.NOM" lemma="|utveckling|" lex="|utveckling..nn.1|" sense="|utveckling..1:1.000|utveckling..2:0.000|" ref="27" dephead="26" deprel="HD">utvecklingen</w>
            <w pos="PP" msd="PP" lemma="|av|" lex="|av..pp.1|" sense="|av..1:-1.000|" ref="28" dephead="25" deprel="ET">av</w>
            <w pos="DT" msd="DT.UTR.SIN.DEF" lemma="|den|en|" lex="|den..pn.1|en..al.1|" sense="|den..2:-1.000|den..1:-1.000|en..2:-1.000|" ref="29" dephead="31" deprel="DT">den</w>
            <w pos="JJ" msd="JJ.POS.UTR+NEU.SIN.DEF.NOM" lemma="|modern|" lex="|modern..av.1|" sense="|modern..1:-1.000|" ref="30" dephead="31" deprel="AT">moderna</w>
            <w pos="NN" msd="NN.UTR.SIN.DEF.NOM" lemma="|verkstadsindustri|" lex="|verkstadsindustri..nn.1|" sense="|verkstadsindustri..1:-1.000|" ref="31" dephead="28" deprel="PA">verkstadsindustrin</w>
            <w pos="MAD" msd="MAD" lemma="|" lex="|" sense="|" ref="32" dephead="15" deprel="IP">.</w>
            </sentence>
            </paragraph>
            </corpus>
                '''

    # def run(self, result=None):
    #    # if type(self) is not SparvCorpusTestCase:
    #    super(SparvCorpusTestCase, self).run(result)

    # def tearDown(self):
    #    # remove files etc....
    #    pass

    def xtest_void(self):
        pass

    def test_can_extract_original_text(self):
        source = [('test.xml', self.xml_data)]
        stream = SparvCorpusSourceReader(source=source, postags="''", chunk_size=None, lemmatize=False)
        _, tokens = next(iter(stream))
        expected_tokens = ['Föremålet', 'som', 'förutom', 'själva', 'vertikalfräsmaskinen', 'består', 'av', 'ett', 'styrskåp',
            'och', 'en', 'omformare', 'utgör', 'ett', 'viktigt', 'steg', 'i', 'utvecklingen', 'av', 'den', 'moderna', 'verkstadsindustrin']
        self.assertSetEqual(set(expected_tokens), set(tokens))

    def test_extract_all_lemma(self):
        source = [('test.xml', self.xml_data)]
        opts = dict(source=source, chunk_size=None, transforms=[ csr.lower_case_transform(), csr.min_token_size_filter(3) ], lemmatize=True)
        expected_tokens = ['föremål', 'som', 'förutom', 'själv', 'vertikalfräsmaskinen', 'bestå', 'styrskåp', 'och',
            'omformare', 'utgöra', 'viktig', 'steg', 'utveckling', 'den', 'modern', 'verkstadsindustri']
        stream = SparvCorpusSourceReader(postags="''", **opts)
        _, tokens = next(iter(stream))
        self.assertSetEqual(set(expected_tokens), set(tokens))

    def test_extract_specific_pos(self):

        source = [('test.xml', self.xml_data)]
        expected_nouns = ['föremål', 'vertikalfräsmaskinen', 'styrskåp', 'omformare', 'steg', 'utveckling', 'verkstadsindustri']
        expected_verbs = ['bestå', 'utgöra' ]

        opts = dict(source=source, chunk_size=None, transforms=[ to_lower ], lemmatize=True)
        stream = SparvCorpusSourceReader(postags="'|NN|'", **opts)
        document, tokens = next(iter(stream))
        print(tokens)
        self.assertEqual('test_01.txt', document)
        self.assertIsNotNone(tokens)
        self.assertSetEqual(set(expected_nouns), set(tokens))

        stream = SparvCorpusSourceReader(postags="'|VB|'", **opts)
        document, tokens = next(iter(stream))

        self.assertEqual('test_01.txt', document)
        self.assertIsNotNone(tokens)
        self.assertSetEqual(set(expected_verbs), set(tokens))

        stream = SparvCorpusSourceReader(postags="'|NN|VB|'", **opts)
        document, tokens = next(iter(stream))

        self.assertEqual('test_01.txt', document)
        self.assertIsNotNone(tokens)
        self.assertSetEqual(set(expected_nouns + expected_verbs), set(tokens))

    def test_lowercase_options(self):
        transforms = [ (lambda tokens: [ x.lower() for x in tokens ]) ]
        xml_data = '''
        <corpus><paragraph>
        <sentence id="ecacfd-e8c6ce">
        <w pos="NN" lemma="|">Mårtensons</w>
        <w pos="NN" lemma="|skapelse|">skapelse</w>
        <w pos="MID" lemma="|">-</w>
        <w pos="JJ" lemma="|svensk|">Svensk</w>
        <w pos="NN" lemma="|">Celluloidindustri</w>
        <w pos="PM" lemma="|AB|">AB</w>
        <w pos="MID" lemma="|">-</w>
        <w pos="VB" lemma="|vara|">är</w>
        <w pos="DT" lemma="|en|">ett</w>
        <w pos="NN" lemma="|exempel|">exempel</w>
        <w pos="PP" lemma="|på|">på</w>
        <w pos="JJ" lemma="|småländsk|">småländsk</w>
        <w pos="NN" lemma="|företagaranda|">företagaranda</w>
        <w pos="MAD" lemma="|" >.</w>
        </sentence>
        </paragraph></corpus>
        '''
        expected_tokens = ['Mårtensons', 'skapelse', 'Celluloidindustri', 'exempel', 'företagaranda']
        source = [('test.xml', xml_data)]
        stream = SparvCorpusSourceReader(source=source, postags="'|NN|'", chunk_size=None)
        _, tokens = next(iter(stream))
        self.assertSetEqual(set(expected_tokens), set(tokens))

        stream = SparvCorpusSourceReader(source=source, postags="'|NN|'", chunk_size=None, transforms=transforms)
        _, tokens = next(iter(stream))
        self.assertSetEqual(set(map(lambda x: x.lower(), expected_tokens)), set(tokens))

    def test_pos_extract_of_larger_file(self):
        filename = '1987_article_08.xml'
        filepath = join_test_data_path(filename)
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        source = [(filename, content)]
        pos_tags = '|PM|'
        stream = SparvCorpusSourceReader(source=source, postags="'{}'".format(pos_tags), chunk_size=9999, transforms=[ to_lower ])
        _, tokens = next(iter(stream))
        self.assertSetEqual(197, len(tokens))

    def test_temp_hyphenation(self):
        from corpora.corpus_source_reader import remove_hyphens
        test_data = u'''
information-
teknik är
    informations-
    teknik är
    informationsteknik är
informations-teknik är
        '''
        expected_result = u'''
informationteknik
 är
    informationsteknik
 är
    informationsteknik är
informations-teknik är
        '''
        service = AnnotateService()
        result = remove_hyphens(test_data)
        self.assertEqual(expected_result, result)

    def test_sparv_corpus(self):

        xml_data = '''
        <corpus><paragraph>
        <sentence id="xxx">
        <w pos="NN"  lemma="|">Humlab</w>
        <w pos="VB"  lemma="|vara|">är</w>
        <w pos="DT"  lemma="|en|">ett</w>
        <w pos="NN"  lemma="|exempel|">exempel</w>
        <w pos="PP"  lemma="|på|">på</w>
        <w pos="DT"  lemma="|en|">ett</w>
        <w pos="NN"  lemma="|arbetsenhet|">arbetsenhet</w>
        <w pos="HP"  lemma="|">som</w>
        <w pos="JJ"  lemma="|digital|">digital</w>
        <w pos="VB"  lemma="|utför|">utför</w>
        <w pos="NN"  lemma="|forskning|">forskning</w>
        <w pos="MAD" lemma="|" >.</w>
        </sentence>
        <sentence id="xxx">
        <w pos="NN"  lemma="|">Humlab</w>
        <w pos="VB"  lemma="|vara|">är</w>
        <w pos="DT"  lemma="|en|">en</w>
        <w pos="NN"  lemma="|arbetsenhet|">arbetsenhet</w>
        <w pos="KN"  lemma="|och|">och</w>
        <w pos="AB"  lemma="|inte|">inte</w>
        <w pos="DT"  lemma="|en|">en</w>
        <w pos="NN"  lemma="|centrumbildning|">centrumbildning</w>
        <w pos="MAD" lemma="|" >.</w>
        </sentence>
        </paragraph></corpus>
        '''

        expected_tokens = ['humlab', 'exempel', 'arbetsenhet', 'forskning', 'centrumbildning']

        source = [ ('test.xml', xml_data)]
        transforms = [ csr.lower_case_transform(), csr.min_token_size_filter(3) ]
        stream = SparvCorpusSourceReader(source=source, postags="'|NN|'", chunk_size=None, transforms=transforms, lemmatize=True)

        corpus = SparvTextCorpus(stream=stream)
        self.assertIsNotNone(corpus)
        document, tokens = next(iter(stream))
        self.assertIsNotNone(tokens)
        self.assertEqual('test_01.txt', document)
        self.assertSetEqual(set(expected_tokens), set(tokens))

        self.assertEqual(len(expected_tokens), len(corpus.dictionary.token2id.keys()))
        self.assertEqual(len(corpus.dictionary.token2id.keys()), len(corpus.dictionary.id2token.keys()))

        temp_corpus_filename = generate_temp_filename('corpus.mm')
        temp_dictionary_filename = generate_temp_filename('corpus.dict.gz')

        corpora.MmCorpus.serialize(temp_corpus_filename, corpus)
        corpus.dictionary.save(temp_dictionary_filename)

        loaded_dictionary = corpora.Dictionary.load(temp_dictionary_filename)
        loaded_corpus = corpora.MmCorpus(temp_corpus_filename)

        self.assertDictEqual(corpus.dictionary.token2id, loaded_dictionary.token2id)
        self.assertDictEqual(corpus.dictionary.id2token, loaded_dictionary.id2token)

        doc0_expected = set((corpus.dictionary[x], y) for x, y in next(iter(corpus)))
        doc0_stored = set((loaded_dictionary[x], y) for x, y in next(iter(loaded_corpus)))

        self.assertSetEqual(doc0_expected, doc0_stored)
        os.remove(temp_corpus_filename)
        os.remove(temp_dictionary_filename)

    def test_tagged_token_read(self):

        xml_data = '''
        <corpus><paragraph>
        <sentence id="xxx">
        <w pos="NN"  lemma="|">Humlab</w>
        <w pos="VB"  lemma="|vara|">är</w>
        <w pos="DT"  lemma="|en|">ett</w>
        <w pos="NN"  lemma="|exempel|">exempel</w>
        <w pos="PP"  lemma="|på|">på</w>
        <w pos="DT"  lemma="|en|">ett</w>
        <w pos="NN"  lemma="|arbetsenhet|">arbetsenhet</w>
        <w pos="HP"  lemma="|">som</w>
        <w pos="JJ"  lemma="|digital|">digital</w>
        <w pos="VB"  lemma="|utför|">utför</w>
        <w pos="NN"  lemma="|forskning|">forskning</w>
        <w pos="MAD" lemma="|" >.</w>
        </sentence>
        <sentence id="xxx">
        <w pos="NN"  lemma="|">Humlab</w>
        <w pos="VB"  lemma="|vara|">är</w>
        <w pos="DT"  lemma="|en|">en</w>
        <w pos="NN"  lemma="|arbetsenhet|">arbetsenhet</w>
        <w pos="KN"  lemma="|och|">och</w>
        <w pos="AB"  lemma="|inte|">inte</w>
        <w pos="DT"  lemma="|en|">en</w>
        <w pos="NN"  lemma="|centrumbildning|">centrumbildning</w>
        <w pos="MAD" lemma="|" >.</w>
        </sentence>
        </paragraph></corpus>
        '''

        expected_tokens = ['humlab_nn', 'är_vb', 'ett_dt', 'exempel_nn', 'på_pp', 'ett_dt', 'arbetsenhet_nn', 'som_hp', 'digital_jj', 'utför_vb',
            'forskning_nn', 'humlab_nn', 'är_vb', 'en_dt', 'arbetsenhet_nn', 'och_kn', 'inte_ab', 'en_dt', 'centrumbildning_nn']

        source = [ ('test.xml', xml_data)]
        transforms = [ csr.lower_case_transform() ]
        stream = SparvCorpusSourceReader(source=source, postags="", chunk_size=None, transforms=transforms, lemmatize=False, append_pos="_" )

        corpus = SparvTextCorpus(stream=stream)
        self.assertIsNotNone(corpus)

        _, tokens = next(iter(stream))

        self.assertIsNotNone(tokens)
        self.assertSetEqual(set(expected_tokens), set(tokens))

    def test_sparv_corpus_stats(self):

        from corpora.corpus_statistics import CorpusPoSStatistics, SUC_POS_TAGS
        import re

        filepath = join_test_data_path('.\\temp\\daedalus_articles_pos_xml_1931-2017.zip')

        column_functions = [
            ('year', lambda df: df.filename.apply(lambda x: int(re.search(r'daedalus_volume_(\d{4})', x).group(1)))),
            ('article', lambda df: df.filename.apply(lambda x: int(re.search(r'article_(\d{2})', x).group(1)))),
            ('segment', lambda df: df.filename.apply(lambda x: int(re.search(r'(\d{2})\.txt', x).group(1))))
        ]

        transforms = [ csr.lower_case_transform(), csr.remove_empty_filter() ]
        opts = dict(transforms=transforms, postags='', lemmatize=False, append_pos="_", ignores="")
        stream = SparvCorpusSourceReader(source=ZipReader(filepath, '*.xml'), **opts)
        df = CorpusPoSStatistics(tag_set=SUC_POS_TAGS).generate(stream, column_functions=column_functions)
        df.to_excel('daedalus_full_pos_stats.xlsx')

        transforms = [ csr.lower_case_transform(), csr.min_token_size_filter(3), csr.remove_empty_filter(), csr.stopwords_filter('swedish') ]
        opts = dict(transforms=transforms, postags='|NN|PM|', lemmatize=True, append_pos="_")
        stream = SparvCorpusSourceReader(source=ZipReader(filepath, '*.xml'), **opts)
        df = CorpusPoSStatistics(tag_set=SUC_POS_TAGS).generate(stream, column_functions=column_functions)
        df.to_excel('daedalus_nn_pm_pos_lemma_nostop_min3_stats.xlsx')

        print(df.columns)

    def test_zip_archive(self):

        filename = '.\\temp\\daedalus_articles_pos_xml_1931-2017.zip'
        filepath = join_test_data_path(filename)
        source = ZipReader(filepath, '*.xml')
        pos_delimiter = "_"
        #for document, tokens in source:
        #    print("{}: {}".format(document, len(tokens)))

        import collections
        import pandas as pd
        import re

        stream = SparvCorpusSourceReader(
            source=source,
            transforms=[ lambda tokens: [ x.lower()  for x in tokens] ],
            postags="''",
            chunk_size=None,
            lemmatize=False,
            append_pos=pos_delimiter,
            ignores=""
        )
        pos_tags = {

            'AB': 0, #  'Adverb',
            'DT': 0, #  'Determiner',
            'HA': 0, #  'WH-adverb',
            'HD': 0, #  'WH-determiner',
            'HP': 0, #  'WH-pronoun',
            'HS': 0, #  'WH-possessive',
            'IE': 0, #  'Infinitival marker',
            'IN': 0, #  'Interjection',
            'JJ': 0, #  'Adjective',
            'KN': 0, #  'Coordinating conjunction',
            'NN': 0, #  'Noun',
            'PC': 0, #  'Participle',
            'PL': 0, #  'Particle',
            'PM': 0, #  'Proper Noun',
            'PN': 0, #  'Pronoun',
            'PP': 0, #  'Preposition',
            'PS': 0, #  'Possessive pronoun',
            'RG': 0, #  'Cardinal number',
            'RO': 0, #  'Ordinal number',
            'SN': 0, #  'Subordinating conjunction',
            'VB': 0, #  'Verb',
            'UO': 0, #  'Foreign word',

            'MAD': 0, #  'Major delimiter',
            'MID': 0, #  'Minor delimiter',
            'PAD': 0, #  'Pairwise delimiter',
            '???': 0
        }

        pos_statistics = []
        pos_total_counter = collections.Counter()
        for document, tokens in stream:

            counter = collections.Counter([ x.split(pos_delimiter)[-1].upper() if pos_delimiter in x else '???' for x in tokens ])
            pos_total_counter.update(counter)

            counter_dict = dict(counter)

            pos_counts = extend(pos_tags, { k: v for k, v in counter_dict.items() if k in pos_tags.keys() })
            other_counts = [ k for k in counter_dict.keys() if k not in pos_tags.keys() ]

            if len(other_counts) > 0:
                logger.warning('Warning strange PoS tags: File %s, tags %s', document, other_counts)

            pos_statistics.append(extend(pos_counts, filename=document))

        #v = {k: [dic[k] for dic in LD] for k in LD[0]}
        df = pd.DataFrame(pos_statistics)
        df['year'] = df.filename.apply(lambda x: int(re.search(r'daedalus_volume_(\d{4})', x).group(1)))
        df['article'] = df.filename.apply(lambda x: int(re.search(r'article_(\d{2})', x).group(1)))
        df['segment'] = df.filename.apply(lambda x: int(re.search(r'(\d{2})\.txt', x).group(1)))
        df.to_excel("stats.xlsx")

        #corpus = SparvTextCorpus(stream, prune_at=2000000)

        #for document, tokens in corpus:
        #    print("{}: {}".format(document, len(tokens)))

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
