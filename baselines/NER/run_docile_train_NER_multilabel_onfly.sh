#!/bin/sh

TIMESTAMP=$(date +"%Y%m%d_%H%M_%S")

# GPU="0"
# GPU="1"
# GPU="2"
# GPU="3"
# GPU="4"
# GPU="5"
GPU="6"
# GPU="7"


# DATASET="docile221221-0"
DATASET="docile221221-0_synthetic"
# TRAIN_DATASET="--hgdataset_dir_train /storage/table_extraction/datasets/${DATASET}/fullpage_multilabel/NER_train"
# VAL_DATASET="--hgdataset_dir_val /storage/table_extraction/datasets/${DATASET}/fullpage_multilabel/NER_val"
# USE_ARROW="--arrow_format"
# STORE_PREPROCESSED=""
# LOAD_PREPROCESSED=""
#
# USE_ARROW=""
# STORE_PREPROCESSED="--store_preprocessed /storage/table_extraction/datasets/docile221221-0/fullpage_multilabel_v2"
# LOAD_PREPROCESSED="--load_from_preprocessed /storage/table_extraction/datasets/docile221221-0/fullpage_multilabel_v2"
# STORE_PREPROCESSED="--store_preprocessed /storage/table_extraction/datasets/docile221221-0/fullpage_multilabel_v3"
# LOAD_PREPROCESSED="--load_from_preprocessed /storage/table_extraction/datasets/docile221221-0/fullpage_multilabel_v3"

DATA="--dataset_name ${DATASET} --docile_path /storage/pif_documents/dataset_exports/docile221221-0/ --on_fly_dataset --dataset_split synthetic "
# TRAIN_PARAMS="--train_bs 32 --test_bs 32 --save_total_limit 3 --weight_decay 0.001 --lr 2e-5 --num_epochs 150 --gradient_accumulation_steps 1 --warmup_ratio 0 --dataloader_num_workers 8"
TRAIN_PARAMS="--train_bs 32 --test_bs 32 --save_total_limit 3 --weight_decay 0.001 --lr 2e-5 --num_epochs 30 --gradient_accumulation_steps 1 --warmup_ratio 0 --dataloader_num_workers 8"
MODEL="--model_name roberta-base --use_roberta --stride 0 --use_BIO_format "
# MODEL="--model_name roberta-base --use_roberta --stride 0 --use_BIO_format --stride 128 "

# OUT_DIR="/storage/table_extraction/trainings/fullpage_multilabel/${DATASET}/RoBERTa_base_gas4_wr_0_5/${TIMESTAMP}"
# OUT_DIR="/storage/table_extraction/trainings/fullpage_multilabel/${DATASET}/RoBERTa_base_gas4_wr01_stride_128/${TIMESTAMP}"
OUT_DIR="/storage/table_extraction/trainings/fullpage_multilabel/${DATASET}/RoBERTa_base/${TIMESTAMP}"
mkdir -p ${OUT_DIR}
LOG="${OUT_DIR}/log_train.txt"

OTHER_PARAMS="--tag_everything --report_all_metrics ${MODEL} "
PARAMS="${TRAIN_PARAMS} ${SAVE_DATASET} ${USE_ARROW} ${OTHER_PARAMS} "
COMMON_PARAMS="${LOAD_PREPROCESSED} ${STORE_PREPROCESSED} ${DATA} --output_dir ${OUT_DIR} ${PARAMS} "

echo "-----------"
echo "Parameters:"
echo "-----------"
echo $COMMON_PARAMS
echo "-----------"
echo "Running TF_FORCE_GPU_ALLOW_GROWTH=\"true\" CUDA_VISIBLE_DEVICES=${GPU} poetry run python /app/baselines/NER/docile_train_NER_multilabel_onfly.py ${COMMON_PARAMS} 2>&1 | tee ${LOG}"
echo "-----------"
echo "==========="

TF_FORCE_GPU_ALLOW_GROWTH="true" CUDA_VISIBLE_DEVICES=${GPU} poetry run python /app/baselines/NER/docile_train_NER_multilabel_onfly.py ${COMMON_PARAMS} 2>&1 | tee ${LOG}
