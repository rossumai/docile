#!/bin/bash

# You can use this script to run NER baselines training. All of the trainings below use either
# publicly available checkpoints (e.g., roberta-base or microsoft/layoutlmv3-base) or checkpoints
# that can be produced with this script (some of them are provided with the docile dataset). To run
# from your checkpoint instead, change the `--model_name` parameter. E.g., to train
# "roberta_ours_with_synthetic_pretraining" from scratch, you should
#  * run "roberta_pretraining",
#  * change the path of `--model_name` in "roberta_ours_synthetic_pretraining" and run it,
#  * change the path of `--model_name` in "roberta_ours_with_synthetic_pretraining" and run it.

set -euo pipefail

# Set GPU device number to use, enforced with CUDA_VISIBLE_DEVICES=${GPU}
GPU="0"

# Choose for which NER baseline you want to run the training. You can run multiple trainings
# consecutively by separating them by space but beware that they run in the order in which they are
# listed at the bottom of this file, not by the order in the `run` list.
run="roberta_base"

TIMESTAMP=$(date +"%Y%m%d_%H%M_%S")
OUTPUT_DIR_PREFIX="/app/data/baselines/trainings"

NER_SCRIPTS_DIR="/app/baselines/NER"
CHECKPOINTS_DIR="/app/data/baselines/checkpoints"

# Common parameters for all trainings with exception of roberta_pretraining
DATA="--dataset_name docile --docile_path /app/data/docile/"
PREPROCESSED_DATASET_DIR="/app/data/baselines/preprocessed_dataset"
USE_PREPROCESSED="--load_from_preprocessed ${PREPROCESSED_DATASET_DIR} --store_preprocessed ${PREPROCESSED_DATASET_DIR}"
OTHER_COMMON_PARAMS="--save_total_limit 3 --weight_decay 0.001 --lr 2e-5 --dataloader_num_workers 8 --use_BIO_format --tag_everything --report_all_metrics --stride 0"
COMMON_PARAMS="${DATA} ${USE_PREPROCESSED} ${OTHER_COMMON_PARAMS}"

function run_training() {
  cmd=$1
  output_dir="${OUTPUT_DIR_PREFIX}/$2"
  shift ; shift
  params="$@ --output_dir ${output_dir}"

  mkdir -p ${output_dir}
  log="${output_dir}/log_train.txt"

  training_cmd="TF_FORCE_GPU_ALLOW_GROWTH=\"true\" CUDA_VISIBLE_DEVICES=${GPU} poetry run python ${NER_SCRIPTS_DIR}/${cmd} ${params} 2>&1 | tee ${log}"

  echo "-----------"
  echo "Parameters:"
  echo "-----------"
  echo ${params}
  echo "-----------"
  echo "Running ${training_cmd}"
  echo "-----------"
  echo "==========="

  eval ${training_cmd}
}


CMD_ROBERTA_PRETRAIN="docile_pretrain_BERT_onfly.py"
CMD_ROBERTA="docile_train_NER_multilabel.py"
CMD_LAYOUTLMV3="docile_train_NER_multilabel_layoutLMv3.py"


single_run="roberta_pretraining"
if [[ " ${run} " =~ " ${single_run} " ]]; then
  data_params="--docile_path /app/data/docile/ --split unlabeled"
  train_params="--per_device_train_batch_size 64 --learning_rate 1e-4 --max_steps 5_000_000 --warmup_steps 5000 --gradient_accumulation_steps 4 --dataloader_num_workers 16"
  model="--model_config_id roberta-base --re_order_ocr_boxes"
  all_params="${data_params} ${train_params} ${model}"
  output_dir="${single_run}/${TIMESTAMP}"
  run_training ${CMD_ROBERTA_PRETRAIN} ${output_dir} ${all_params}
fi

single_run="roberta_base"
if [[ " ${run} " =~ " ${single_run} " ]]; then
  train_params="--train_bs 32 --test_bs 32 --num_epochs 1500 --gradient_accumulation_steps 4 --warmup_ratio 1.0"
  model="--model_name roberta-base --use_roberta"
  all_params="${COMMON_PARAMS} ${train_params} ${model}"
  output_dir="${single_run}/${TIMESTAMP}"
  run_training ${CMD_ROBERTA} ${output_dir} ${all_params}
fi

single_run="roberta_ours"
if [[ " ${run} " =~ " ${single_run} " ]]; then
  train_params="--train_bs 32 --test_bs 32 --num_epochs 1000 --gradient_accumulation_steps 1 --warmup_ratio 0"
  model="--model_name ${CHECKPOINTS_DIR}/roberta_pretraining_50000 --use_roberta"
  all_params="${COMMON_PARAMS} ${train_params} ${model}"
  output_dir="${single_run}/${TIMESTAMP}"
  run_training ${CMD_ROBERTA} ${output_dir} ${all_params}
fi

single_run="layoutlmv3_base"
if [[ " ${run} " =~ " ${single_run} " ]]; then
  train_params="--train_bs 32 --test_bs 32 --num_epochs 1500 --gradient_accumulation_steps 1 --warmup_ratio 0"
  model="--model_name microsoft/layoutlmv3-base"
  all_params="${COMMON_PARAMS} ${train_params} ${model}"
  output_dir="${single_run}/${TIMESTAMP}"
  run_training ${CMD_LAYOUTLMV3} ${output_dir} ${all_params}
fi

single_run="layoutlmv3_ours"
if [[ " ${run} " =~ " ${single_run} " ]]; then
  train_params="--train_bs 16 --test_bs 16 --num_epochs 1000 --gradient_accumulation_steps 1 --warmup_ratio 0"
  model="--model_name microsoft/layoutlmv3-base --pretrained_weights ${CHECKPOINTS_DIR}/layoutlmv3_pretraining.ckpt"
  all_params="${COMMON_PARAMS} ${train_params} ${model}"
  output_dir="${single_run}/${TIMESTAMP}"
  run_training ${CMD_LAYOUTLMV3} ${output_dir} ${all_params}
fi

single_run="roberta_base_synthetic_pretraining"  # 30 epochs on synthetic data only
if [[ " ${run} " =~ " ${single_run} " ]]; then
  data_params="--split synthetic"
  train_params="--train_bs 16 --test_bs 16 --num_epochs 30 --gradient_accumulation_steps 1 --warmup_ratio 0"
  model="--model_name roberta-base --use_roberta"
  all_params="${COMMON_PARAMS} ${data_params} ${train_params} ${model}"
  output_dir="${single_run}/${TIMESTAMP}"
  run_training ${CMD_ROBERTA} ${output_dir} ${all_params}
fi

single_run="roberta_ours_synthetic_pretraining"  # 30 epochs on synthetic data only
if [[ " ${run} " =~ " ${single_run} " ]]; then
  data_params="--split synthetic"
  train_params="--train_bs 16 --test_bs 16 --num_epochs 30 --gradient_accumulation_steps 1 --warmup_ratio 0"
  model="--model_name ${CHECKPOINTS_DIR}/roberta_pretraining_50000 --use_roberta"
  all_params="${COMMON_PARAMS} ${data_params} ${train_params} ${model}"
  output_dir="${single_run}/${TIMESTAMP}"
  run_training ${CMD_ROBERTA} ${output_dir} ${all_params}
fi

single_run="layoutlmv3_ours_synthetic_pretraining"  # 30 epochs on synthetic data only
if [[ " ${run} " =~ " ${single_run} " ]]; then
  data_params="--split synthetic"
  train_params="--train_bs 16 --test_bs 16 --num_epochs 30 --gradient_accumulation_steps 1 --warmup_ratio 0"
  model="--model_name microsoft/layoutlmv3-base --pretrained_weights ${CHECKPOINTS_DIR}/layoutlmv3_pretraining.ckpt"
  all_params="${COMMON_PARAMS} ${data_params} ${train_params} ${model}"
  output_dir="${single_run}/${TIMESTAMP}"
  run_training ${CMD_LAYOUTLMV3} ${output_dir} ${all_params}
fi

single_run="roberta_base_with_synthetic_pretraining"
if [[ " ${run} " =~ " ${single_run} " ]]; then
  train_params="--train_bs 8 --test_bs 8 --num_epochs 1500 --gradient_accumulation_steps 8 --warmup_ratio 0"
  model="--model_name ${CHECKPOINTS_DIR}/roberta_base_synthetic_pretraining_187500 --use_roberta"
  all_params="${COMMON_PARAMS} ${train_params} ${model}"
  output_dir="${single_run}/${TIMESTAMP}"
  run_training ${CMD_ROBERTA} ${output_dir} ${all_params}
fi

single_run="roberta_ours_with_synthetic_pretraining"
if [[ " ${run} " =~ " ${single_run} " ]]; then
  train_params="--train_bs 8 --test_bs 8 --num_epochs 1500 --gradient_accumulation_steps 4 --warmup_ratio 0"
  model="--model_name ${CHECKPOINTS_DIR}/roberta_ours_synthetic_pretraining_187500 --use_roberta"
  all_params="${COMMON_PARAMS} ${train_params} ${model}"
  output_dir="${single_run}/${TIMESTAMP}"
  run_training ${CMD_ROBERTA} ${output_dir} ${all_params}
fi

single_run="layoutlmv3_ours_with_synthetic_pretraining"
if [[ " ${run} " =~ " ${single_run} " ]]; then
  train_params="--train_bs 16 --test_bs 16 --num_epochs 1500 --gradient_accumulation_steps 1 --warmup_ratio 0"
  model="--model_name microsoft/layoutlmv3-base --pretrained_weights ${CHECKPOINTS_DIR}/layoutlmv3_ours_synthetic_pretraining_187500"
  all_params="${COMMON_PARAMS} ${train_params} ${model}"
  output_dir="${single_run}/${TIMESTAMP}"
  run_training ${CMD_LAYOUTLMV3} ${output_dir} ${all_params}
fi


# Not presented in the dataset paper
single_run="roberta_base_with_2d_embedding"
if [[ " ${run} " =~ " ${single_run} " ]]; then
  train_params="--train_bs 32 --test_bs 32 --num_epochs 1500 --gradient_accumulation_steps 4 --warmup_ratio 1.0"
  model="--model_name roberta-base --use_roberta --use_new_2D_pos_emb --pos_emb_dim 6500"
  all_params="${COMMON_PARAMS} ${train_params} ${model}"
  output_dir="${single_run}/${TIMESTAMP}"
  run_training ${CMD_ROBERTA} ${output_dir} ${all_params}
fi
