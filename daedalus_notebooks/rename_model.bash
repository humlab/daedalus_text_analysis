#!/bin/bash
BASENAME=topics_300chunk1000_iterations_2000_lowercase_ldamallet
NEW_BASENAME=SOU_topics_300_chunks_1000_iterations_2000_lowercase_ldamallet
mv ./${BASENAME}/result_${BASENAME}_dictionary.csv ./${BASENAME}/result_${NEW_BASENAME}_dictionary.csv
mv ./${BASENAME}/result_${BASENAME}_doc_topic_weights.csv ./${BASENAME}/result_${NEW_BASENAME}_doc_topic_weights.csv
mv ./${BASENAME}/result_${BASENAME}_documents.csv ./${BASENAME}/result_${NEW_BASENAME}_documents.csv
mv ./${BASENAME}/result_${BASENAME}_doc_topic_weights.csv ./${BASENAME}/result_${NEW_BASENAME}_doc_topic_weights.csv
mv ./${BASENAME}/result_${BASENAME}_topickeys.csv ./${BASENAME}/result_${NEW_BASENAME}_topickeys.csv
mv ./${BASENAME}/result_${BASENAME}_topic_tokens.csv ./${BASENAME}/result_${NEW_BASENAME}_topic_tokens.csv
mv ./${BASENAME}/result_${BASENAME}_topic_token_weights.csv ./${BASENAME}/result_${NEW_BASENAME}_topic_token_weights.csv
mv ./${BASENAME}/result_${BASENAME}.xlsx ./${BASENAME}/result_${NEW_BASENAME}.xlsx
mv ./${BASENAME} ./${NEW_BASENAME}
