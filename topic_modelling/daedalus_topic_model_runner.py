
from . import compute

mallet_path = 'C:\\Usr\\mallet-2.0.8'

if __name__ == "__main__":

    source = '../data/daedalus_sparv_result_article_xml_20171122.zip'
    '''
    See https://spraakbanken.gu.se/korp/markup/msdtags.html for description of MSD-tag set
    '''

    options = [

        { 'lda_engine': 'LdaModel', 'engine_lda_options': { "num_topics": 50, "iterations": 2000, 'chunksize': 10000, 'passes': 2 }},
        { 'lda_engine': 'LdaMallet', 'lda_options': { "num_topics": 50, "iterations": 2000 }, 'engine_path': mallet_path  },

        { 'lda_engine': 'LdaMallet', 'lda_options': { "num_topics": 100, "iterations": 2000 }, 'engine_path': mallet_path  },
        { 'lda_engine': 'LdaModel', 'lda_options': { "num_topics": 100, "iterations": 2000, 'chunksize': 10000, 'passes': 2  } },

        { 'lda_engine': 'LdaModel', 'lda_options': { "num_topics": 150, "iterations": 2000, 'chunksize': 10000, 'passes': 2  } },
        { 'lda_engine': 'LdaMallet', 'lda_options': { "num_topics": 150, "iterations": 2000 }, 'engine_path': mallet_path  }
        # { 'lda_engine': 'LdaModel', 'lda_options': { "num_topics": 50, "iterations": 999, 'chunksize': 10000, 'passes': 2 } }
    ]

    compute(source=source, options=options)
