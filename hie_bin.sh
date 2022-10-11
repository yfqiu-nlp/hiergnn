TASK=cnndm-ssplit
DICT=checkpoints/dict.source.txt
fairseq-preprocess \
    --source-lang "source" \
    --target-lang "target" \
    --trainpref "${TASK}/train.bpe" \
    --validpref "${TASK}/val.bpe" \
    --destdir "${TASK}-bin/" \
    --workers 60 \
    --srcdict $DICT \
    --tgtdict $DICT;
