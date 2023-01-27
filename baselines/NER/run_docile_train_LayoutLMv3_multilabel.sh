#!/bin/sh

TIMESTAMP=$(date +"%Y%m%d_%H%M_%S")

# GPU="0"
GPU="1"
# GPU="2"
# GPU="3"
# GPU="4"
# GPU="5"
# GPU="6"
# GPU="7"


DATASET="docile221221-0"
TRAIN_DATASET="--hgdataset_dir_train /storage/table_extraction/datasets/${DATASET}/fullpage_multilabel/NER_LayoutLMv3_NER_train/"
VAL_DATASET="--hgdataset_dir_val /storage/table_extraction/datasets/${DATASET}/fullpage_multilabel/NER_LayoutLMv3_NER_val/"
# USE_ARROW="--save_datasets_in_arrow_format /storage/table_extraction/datasets/${DATASET}/fullpage_multilabel "
USE_ARROW="--arrow_format"
STORE_PREPROCESSED=""
# LOAD_PREPROCESSED=""
# STORE_PREPROCESSED="--store_preprocessed /storage/table_extraction/datasets/docile221221-0/fullpage_multilabel"
LOAD_PREPROCESSED="--load_from_preprocessed /storage/table_extraction/datasets/docile221221-0/fullpage_multilabel"
#

DATA="--dataset_name ${DATASET} --docile_path /storage/pif_documents/dataset_exports/docile221221-0/ ${TRAIN_DATASET} ${VAL_DATASET}"
# TRAIN_PARAMS="--train_bs 32 --test_bs 32 --save_total_limit 3 --weight_decay 0.001 --lr 2e-5 --num_epochs 1500 --gradient_accumulation_steps 1 --warmup_ratio 0.0 --dataloader_num_workers 8"
TRAIN_PARAMS="--train_bs 32 --test_bs 32 --save_total_limit 3 --weight_decay 0.001 --lr 2e-5 --num_epochs 1500 --gradient_accumulation_steps 1 --warmup_ratio 0.25 --dataloader_num_workers 8"
MODEL="--model_name microsoft/layoutlmv3-base --stride 0 --use_BIO_format "

# OUT_DIR="/storage/table_extraction/trainings/fullpage_multilabel/${DATASET}/LayoutLMv3/${TIMESTAMP}"
OUT_DIR="/storage/table_extraction/trainings/fullpage_multilabel/${DATASET}/LayoutLMv3_wr_0_25/${TIMESTAMP}"
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
echo "Running TF_FORCE_GPU_ALLOW_GROWTH=\"true\" CUDA_VISIBLE_DEVICES=${GPU} poetry run python /app/baselines/NER/docile_train_NER_multilabel_layoutLMv3.py ${COMMON_PARAMS} 2>&1 | tee ${LOG}"
echo "-----------"
echo "==========="

TF_FORCE_GPU_ALLOW_GROWTH="true" CUDA_VISIBLE_DEVICES=${GPU} poetry run python /app/baselines/NER/docile_train_NER_multilabel_layoutLMv3.py ${COMMON_PARAMS} 2>&1 | tee ${LOG}
