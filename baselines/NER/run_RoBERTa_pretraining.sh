#!/bin/sh

TIMESTAMP=$(date +"%Y%m%d_%H%M_%S")

# GPU=""
# GPU="1"
GPU="7"

# OUT_PATH="/storage/table_extraction/pretrain_roberta/v2"
# OUT_PATH="/storage/table_extraction/pretrain_roberta/v2_with_reOrdering"
OUT_PATH="/storage/table_extraction/pretrain_roberta/v2b_with_reOrdering"
DOCILE="--docile_path /storage/pif_documents/dataset_exports/docile_pretraining_v1_2022_12_22/ --split pretraining-all.json"
# MODEL="--model_config_id roberta-base --per_device_train_batch_size 32 --learning_rate 1e-4  --max_steps 5_000_000 --warmup_steps 5000 "
MODEL="--model_config_id roberta-base --per_device_train_batch_size 64 --learning_rate 1e-4  --max_steps 5_000_000 --warmup_steps 5000  --gradient_accumulation_steps 4 "
# COMMON_PARAMS="--repository_id /storage/table_extraction/pretrain_roberta/ ${DOCILE} ${MODEL} --dataloader_num_workers 16 --repository_id /storage/table_extraction/pretrain_roberta/v2/ "
# COMMON_PARAMS="--repository_id ${OUT_PATH} ${DOCILE} ${MODEL} --dataloader_num_workers 16 "
COMMON_PARAMS="--repository_id ${OUT_PATH} ${DOCILE} ${MODEL} --dataloader_num_workers 16 --re_order_ocr_boxes"

mkdir -p ${OUT_PATH}

LOG="${OUT_PATH}/log_docile_pretraining.txt"

echo $COMMON_PARAMS
echo ""
echo "Running TF_FORCE_GPU_ALLOW_GROWTH=\"true\" CUDA_VISIBLE_DEVICES=${GPU} python /root/ner_for_tables/docile_pretrain_BERT_onfly.py ${COMMON_PARAMS} 2>&1 | tee ${LOG}"
echo ""

TF_FORCE_GPU_ALLOW_GROWTH="true" CUDA_VISIBLE_DEVICES=${GPU} python /root/ner_for_tables/docile_pretrain_BERT_onfly.py ${COMMON_PARAMS} 2>&1 | tee ${LOG}
