#!/bin/bash
DATA_PATH=""
MODEL_PATH=""
CHECKPOINT_INT=""
LOG_FREQ=""
METHOD=""
PCT=""
SEED=""
EARLY_STOPPING="True"
NO_TRAIN_MODEL="False"
MODEL='MLP'

while [[ $# -gt 0 ]]; do
  case $1 in
    --datapath)
      DATA_PATH=$2
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
    --pct)
      PCT=$2
      shift 2
      ;;
    --seed)
      SEED=$2
      shift 2
      ;;
    --method)
      METHOD=$2
      shift 2
      ;;
    --max-steps)
      MAX_STEPS=$2
      shift 2
      ;;
    --early_stopping)
      EARLY_STOPPING=$2
      shift 2
      ;;
    --no_train_model)
      NO_TRAIN_MODEL=$2
      shift 2
      ;;
    --model)
      MODEL=$2
      shift 2
      ;;
    *)
      echo hi
      ;;
  esac
done


echo $DATA_PATH $MODEL_PATH $CHECKPOINT_INT $LOG_FREQ $METHOD $PCT $SEED
DATA_PATH=$DATA_PATH/idx_${PCT}pct_seed${SEED}

echo python -u ssl_in_scg/self_supervision/trainer/masking/train.py --mask_rate 0.5 --model $MODEL --dropout 0.1 --weight_decay 0.01 --lr 0.001 --data_path $DATA_PATH --model_path $MODEL_PATH --checkpoint_interval $CHECKPOINT_INT --log_freq $LOG_FREQ --max_steps $MAX_STEPS --early_stopping $EARLY_STOPPING --no_train_model $NO_TRAIN_MODEL --decoder
python -u ssl_in_scg/self_supervision/trainer/masking/train.py --mask_rate 0.5 --model $MODEL --dropout 0.1 --weight_decay 0.01 --lr 0.001 --data_path $DATA_PATH --model_path $MODEL_PATH --checkpoint_interval $CHECKPOINT_INT --log_freq $LOG_FREQ --max_steps $MAX_STEPS --early_stopping $EARLY_STOPPING --no_train_model $NO_TRAIN_MODEL --decoder