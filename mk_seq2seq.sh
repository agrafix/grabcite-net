#!/bin/bash

set -e

DATASET="tiny"

if [ "$1" == "generate" ]; then
    echo "Generating data for $DATASET"

    python3 seq2seq_data.py
    seq2seq/bin/tools/generate_vocab.py --delimiter " " --max_vocab_size 50000 < datasets/$DATASET.plain.txt > datasets/$DATASET.plain.vocab.txt
    cat datasets/$DATASET.plain.vocab.txt datasets/${DATASET}_allCits.cit.txt > datasets/$DATASET.cit.vocab.txt
fi

if [ "$1" == "train" ]; then
    echo "Training on $DATASET"

    export VOCAB_SOURCE=../datasets/$DATASET.plain.vocab.txt
    export VOCAB_TARGET=../datasets/$DATASET.cit.vocab.txt
    export TRAIN_SOURCES=../datasets/$DATASET.plain.txt
    export TRAIN_TARGETS=../datasets/$DATASET.cit.txt
    export DEV_SOURCES=../datasets/$DATASET.plain.test.txt
    export DEV_TARGETS=../datasets/$DATASET.cit.test.txt

    export TRAIN_STEPS=1000

    export MODEL_DIR=../seq2seq_model/$DATASET/$(date +%F_%R)/
    cd seq2seq

    mkdir -p $MODEL_DIR
    echo "Model will be written to $MODEL_DIR"

    python3 -m bin.train \
      --config_paths="
          ./example_configs/nmt_small.yml,
          ./example_configs/train_seq2seq.yml,
          ./example_configs/text_metrics_bpe.yml" \
      --model_params "
          vocab_source: $VOCAB_SOURCE
          vocab_target: $VOCAB_TARGET" \
      --input_pipeline_train "
        class: ParallelTextInputPipeline
        params:
          source_files:
            - $TRAIN_SOURCES
          target_files:
            - $TRAIN_TARGETS" \
      --input_pipeline_dev "
        class: ParallelTextInputPipeline
        params:
           source_files:
            - $DEV_SOURCES
           target_files:
            - $DEV_TARGETS" \
      --batch_size 32 \
      --train_steps $TRAIN_STEPS \
      --output_dir $MODEL_DIR
fi

if [ "$1" == "predict" ]; then
    if [ -z ${2+x} ]; then
        echo "Missing second parameter describing model id"
        exit 1
    fi
    export MODEL_DIR=../seq2seq_model/$DATASET/$2/
    cd seq2seq

    export PRED_DIR=${MODEL_DIR}/pred
    mkdir -p ${PRED_DIR}

    export VALID_SOURCES=../datasets/$DATASET.plain.valid.txt

    python3 -m bin.infer \
      --tasks "
        - class: DecodeText" \
      --model_dir $MODEL_DIR \
      --input_pipeline "
        class: ParallelTextInputPipeline
        params:
          source_files:
            - $VALID_SOURCES" \
      >  ${PRED_DIR}/predictions.txt

    echo "Wrote to ${PRED_DIR}/predictions.txt"
fi
