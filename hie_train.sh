#!/bin/sh
set -e

TOTAL_NUM_UPDATES=30000
WARMUP_UPDATES=250
LR=3e-05
MAX_TOKENS=2048 
UPDATE_FREQ=4
BART_PATH="bart.large/model.pt"
SAVE_DIR=checkpoints
DATA_BIN=cnndm-ssplit-bin # cnndm

export PYTHONPATH=fairseq
python3 $(which fairseq-train) $DATA_BIN \
    --restore-file $BART_PATH \
    --max-tokens $MAX_TOKENS \
    --task translation \
    --source-lang source --target-lang target \
    --truncate-source \
    --layernorm-embedding \
    --share-all-embeddings \
    --share-decoder-input-output-embed \
    --reset-optimizer --reset-dataloader --reset-meters \
    --required-batch-size-multiple 1 \
    --arch bart_large \
    --criterion label_smoothed_cross_entropy \
    --label-smoothing 0.1 \
    --dropout 0.1 --attention-dropout 0.1 \
    --weight-decay 0.01 --optimizer adam --adam-betas "(0.9, 0.999)" --adam-eps 1e-08 \
    --clip-norm 0.1 \
    --lr-scheduler polynomial_decay --lr $LR --total-num-update $TOTAL_NUM_UPDATES --warmup-updates $WARMUP_UPDATES \
    --fp16 \
    --update-freq $UPDATE_FREQ \
    --skip-invalid-size-inputs-valid-test \
    --save-dir $SAVE_DIR \
    --find-unused-parameters \
    --max-update $TOTAL_NUM_UPDATES \
    --gnn_dp_rate 0.3 \
    --hop_layer 2 \
    

