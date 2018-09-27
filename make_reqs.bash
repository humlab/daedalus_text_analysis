#!/bin/bash

IPYNB_SCRIPTS=./ipynb_python.py
RESULT_FILE=./pipreqs_requirements.txt
# convert notebooks to scripts
find . \( -type d -name ".ipynb_checkpoints" -prune \) -o -type f -name *.ipynb -exec jupyter nbconvert --stdout --to script {} \; > $IPYNB_SCRIPTS

rm -f $RESULT_FILE ${RESULT_FILE}.tmp

pipreqs --force --savepath ${RESULT_FILE}.tmp .

sort ${RESULT_FILE}.tmp > ${RESULT_FILE}

rm -f $IPYNB_SCRIPTS

