
from sparv_annotater import ArchiveAnnotater

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
        }
    }

    service = ArchiveAnnotater(settings=settings, transforms=[lambda x: x.replace('\x0c',' ')])
    source_name = './data/UNTS_1945-1972_extracted_txt.zip'
    target_name = './data/UNTS_1945-1972_pos_xml.zip'
    service.annotate_files_in_archive(source_filename=source_name, target_filename=target_name)

main()
