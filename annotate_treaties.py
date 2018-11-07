
from sparv_annotater import ArchiveAnnotater
from common.utility import filename_add_suffix

def main():
    settings = {
        "corpus": "untitled",
        "lang": "en",
        "textmode": "file",
        "positional_attributes": {
            "lexical_attributes": ["pos", "msd", "lemma"]
        },
        "text_attributes": {
            "readability_metrics": []
        },
        "named_entity_recognition": ["ex", "type", "subtype"]
    }
    # '\xe9', '\xf4', '\xc9', '\xe0', '\xe7', '\x8e', '\xab', '\x83', '\x85', '\xd5', '\xca',

    service = ArchiveAnnotater(settings=settings, transforms=[lambda x: x.replace('\x0c',' ')]) # lambda x: x.replace('\x0c',' '), lambda x: x.replace('\xe9',' ')])

    source_name = './data/treaty_text_corpora_en_20101022.zip'
    target_name = filename_add_suffix(source_name, '_pos_xml')

    service.annotate_files_in_archive(source_filename=source_name, target_filename=target_name)

main()
