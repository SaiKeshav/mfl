#!/bin/bash

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

DATASET=$1
MODEL=$2

echo "Processing ${DATASET}"

for SPLIT in train dev test; do
    for LANG in "source" "target"; do
        # python scripts_mgenre/preprocess_sentencepiece.py --m ${MODEL}/sentence.bpe.model \
        python scripts_mgenre/preprocess_sentencepiece.py --m ${MODEL}/spm_256000.model \
        --inputs ${DATASET}/${SPLIT}.${LANG} \
        --outputs ${DATASET}/${SPLIT}.spm.${LANG} \
        --workers 40
    done
done

fairseq-preprocess \
  --source-lang "source" \
  --target-lang "target" \
  --trainpref ${DATASET}/train.spm \
  --validpref ${DATASET}/dev.spm \
  --testpref ${DATASET}/test.spm \
  --destdir ${DATASET}/bin \
  --thresholdtgt 0 \
  --thresholdsrc 0 \
  --srcdict $MODEL"/dict.source.txt" \
  --tgtdict $MODEL"/dict.target.txt" \
  --workers 40;

  # --srcdict $MODEL"/dict.txt" \
  # --tgtdict $MODEL"/dict.txt" \
