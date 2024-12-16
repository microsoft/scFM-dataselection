#!/bin/bash


PRETRAINED_DIR=""
DATA_PATH=""
MODEL_PATH=""
CHECKPOINT_INT=""
LOG_FREQ=""
# set -e


while [[ $# -gt 0 ]]; do
  case $1 in
    --datapath)
      DATA_PATH=$2
      shift 2
      ;;
    --pretrained-dir)
      PRETRAINED_DIR=$2
      shift 2
      ;;
    --model-path)
      MODEL_PATH=$2
      shift 2
      ;;
    --checkpoint-int)
      CHECKPOINT_INT=$2
      shift 2
      ;;
    --log-freq)
      LOG_FREQ=$2
      shift 2
      ;;
    *)
      echo hi
      ;;
  esac
done

pretrained_model=$(find $PRETRAINED_DIR -name "best_checkpoint_val.ckpt" -print0 | xargs -0 ls -lt | head -n 1 | awk '{print $NF}')
if [ -z "$pretrained_model" ]; then
    echo "File best_checkpoint_val.ckpt not found in $PRETRAINED_DIR."
    exit 1
fi
echo pretrained model $pretrained_model
python -u ssl_in_scg/self_supervision/trainer/classifier/cellnet_mlp.py --version="new_run0" --lr=9e-4 --dropout=0.1 --weight_decay=0.05 --stochastic --pretrained_dir=$pretrained_model --data_path $DATA_PATH --model_path $MODEL_PATH --checkpoint_interval $CHECKPOINT_INT --log_freq $LOG_FREQ

