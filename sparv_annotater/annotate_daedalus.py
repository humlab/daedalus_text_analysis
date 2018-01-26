
from sparv_annotater import ArchiveAnnotater

def main():
    settings = {
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
            "lexical_attributes": ["pos", "msd", "lemma"],  # , "lex", "sense"],
            "compound_attributes": [],
            "dependency_attributes": [ ],  # "ref", "dephead", "deprel"],
            "sentiment": []
        },
        "named_entity_recognition": [],
        "text_attributes": {
            "readability_metrics": []
        }
    }

    service = ArchiveAnnotater(settings)
    source_name = '../data/segmented-yearly-volumes_articles.zip'
    service.annotate_files_in_archive(source_name)

main()
