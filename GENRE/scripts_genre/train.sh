#!/bin/bash

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

langs=ar_AR,cs_CZ,de_DE,en_XX,es_XX,et_EE,fi_FI,fr_XX,gu_IN,hi_IN,it_IT,ja_XX,kk_KZ,ko_KR,lt_LT,lv_LV,my_MM,ne_NP,nl_XX,ro_RO,ru_RU,si_LK,tr_TR,vi_VN,zh_CN

mkdir -p $NAME
rsync -rv --exclude model.pt ../models/fairseq_multilingual_entity_disambiguation/ $NAME

fairseq-train $NAME/bin/ \
    --save-dir $NAME \
    --no-epoch-checkpoints \
    --no-last-checkpoints \
    --no-save-optimizer-state \
    --tensorboard-logdir tensorboard_logs/logs \
    --restore-file ../models/fairseq_multilingual_entity_disambiguation/model.pt \
    --task translation  \
    --max-epoch 125 \
    --reset-meters  \
    --reset-optimizer \
    --reset-lr-scheduler \
    --arch mbart_large \
    --criterion label_smoothed_cross_entropy  \
    --source-lang source  \
    --target-lang target  \
    --truncate-source  \
    --label-smoothing 0.1  \
    --max-tokens 4096  \
    --update-freq 1  \
    --max-update 200000  \
    --required-batch-size-multiple 1  \
    --dropout 0.1  \
    --attention-dropout 0.1  \
    --relu-dropout 0.0  \
    --weight-decay 0.01  \
    --optimizer adam  \
    --adam-betas "(0.9, 0.999)"  \
    --adam-eps 1e-08  \
    --clip-norm 0.1  \
    --lr-scheduler polynomial_decay  \
    --lr 3e-05  \
    --total-num-update 200000  \
    --ddp-backend no_c10d  \
    --num-workers 20  \
    --share-all-embeddings \
    --layernorm-embedding \
    --encoder-normalize-before --decoder-normalize-before \
    --share-decoder-input-output-embed  \
    --skip-invalid-size-inputs-valid-test  \
    --log-format json  \
    --patience 200

