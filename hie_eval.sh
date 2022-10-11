#!/bin/sh
set -e

python3 -m rouge.rouge \
    --target_filepattern=cnndm-ssplit/test.target \
    --prediction_filepattern=test.hypo.cnndm \
    --output_filename=cnndm.scores.csv \
    --use_stemmer=true \
    --split_summaries=true
