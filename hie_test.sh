#!/bin/sh
set -e

export PYTHONPATH=fairseq
python3 summarize.py   \
	--model-dir checkpoints   \
	--model-file checkpoint_best.pt   \
	--src cnndm-ssplit/test.source   \
	--out test.hypo.cnndm \
