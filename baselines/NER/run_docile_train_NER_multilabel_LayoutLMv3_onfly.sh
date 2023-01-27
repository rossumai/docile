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

DATA="--dataset_name ${DATASET} --docile_path /storage/pif_documents/dataset_exports/docile221221-0/ --on_fly_dataset --dataset_split synthetic "
# TRAIN_PARAMS="--train_bs 32 --test_bs 32 --save_total_limit 3 --weight_decay 0.001 --lr 2e-5 --num_epochs 150 --gradient_accumulation_steps 1 --warmup_ratio 0 --dataloader_num_workers 8"
TRAIN_PARAMS="--train_bs 32 --test_bs 32 --save_total_limit 3 --weight_decay 0.001 --lr 2e-5 --num_epochs 30 --gradient_accumulation_steps 1 --warmup_ratio 0 --dataloader_num_workers 8"
# MODEL="--model_name microsoft/layoutlmv3-base --stride 0 --use_BIO_format  "  # HuggingFace init
MODEL="--model_name microsoft/layoutlmv3-base --stride 0 --use_BIO_format --pretrained_weights /storage/table_extraction/pretrained_layoutlmv3_v2.ckpt  "  # Our pretraining init

# pretrained_weights

OUT_DIR="/storage/table_extraction/trainings/fullpage_multilabel/${DATASET}/LayoutLMv3_fromOurPretrained/${TIMESTAMP}"
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
echo "Running TF_FORCE_GPU_ALLOW_GROWTH=\"true\" CUDA_VISIBLE_DEVICES=${GPU} poetry run python /app/baselines/NER/docile_train_NER_multilabel_layoutLMv3_onfly.py ${COMMON_PARAMS} 2>&1 | tee ${LOG}"
echo "-----------"
echo "==========="

TF_FORCE_GPU_ALLOW_GROWTH="true" CUDA_VISIBLE_DEVICES=${GPU} poetry run python /app/baselines/NER/docile_train_NER_multilabel_layoutLMv3_onfly.py ${COMMON_PARAMS} 2>&1 | tee ${LOG}
