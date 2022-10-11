TASK=cnndm-ssplit
PROG=fairseq/examples/roberta/multiprocessing_bpe_encoder.py

for SPLIT in train val
do
    for LANG in source target
    do
	python $PROG \
	       --encoder-json hie_encoder.json \
	       --vocab-bpe vocab.bpe \
	       --inputs "$TASK/$SPLIT.$LANG" \
	       --outputs "$TASK/$SPLIT.bpe.$LANG" \
	       --workers 60 \
	       --keep-empty;
    done
done
