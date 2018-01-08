
from . import compute

if __name__ == "__main__":

    options_list = [
        { 'skip': False, 'id': '_complete_not_segmented', 'input_path': '../data/input/not_segmented/*.zip', 'output_path': '../data/output', 'window': 5, 'sg': 1, 'size': 50, 'min_count': 5, 'iter': 20, 'workers': 10, 'filter_stopwords': False, 'bigram_transformer': False },
        { 'skip': False, 'id': '_segmented_1980-1014', 'input_path': '../data/input/daedalus-segmenterad.zip', 'output_path': '../data/output', 'window': 5, 'sg': 1, 'size': 100, 'min_count': 5, 'iter': 20, 'workers': 10, 'filter_stopwords': False, 'bigram_transformer': False },
        { 'skip': False, 'id': '_segmented_1980-1014', 'input_path': '../data/input/daedalus-segmenterad.zip', 'output_path': '../data/output', 'window': 7, 'sg': 1, 'size': 100, 'min_count': 5, 'iter': 20, 'workers': 10, 'filter_stopwords': False, 'bigram_transformer': False },
        { 'skip': False, 'id': '_segmented_1980-1014', 'input_path': '../data/input/daedalus-segmenterad.zip', 'output_path': '../data/output', 'window': 10, 'sg': 1, 'size': 100, 'min_count': 5, 'iter': 20, 'workers': 10, 'filter_stopwords': False, 'bigram_transformer': False },
        { 'skip': False, 'id': '_complete_not_segmented', 'input_path': '../data/input/not_segmented/*.zip', 'output_path': '../data/output', 'window': 5, 'sg': 1, 'size': 100, 'min_count': 5, 'iter': 20, 'workers': 10, 'filter_stopwords': False, 'bigram_transformer': False },
        { 'skip': False, 'id': '_complete_not_segmented', 'input_path': '../data/input/not_segmented/*.zip', 'output_path': '../data/output', 'window': 5, 'sg': 1, 'size': 150, 'min_count': 5, 'iter': 20, 'workers': 10, 'filter_stopwords': False, 'bigram_transformer': False }
    ]

    for options in options_list:
        compute(options)
        