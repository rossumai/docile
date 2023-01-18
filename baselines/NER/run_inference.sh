#!/bin/bash

set -euo pipefail

# Set GPU device number to use, enforced with CUDA_VISIBLE_DEVICES=${GPU}
GPU="0"

# Call with `./run_inference test` or `./run_inference.sh val`
split=$1

# Choose baselines to run inference for by setting `run="baseline_name1 baseline_name2 ..."` below.
# Beware that they run in the order in which they are listed at the bottom of this file, not by the
# order in the `run` list.
# Some examples:
run_main="roberta_base roberta_ours layoutlmv3_base layoutlmv3_ours"
run_with_detr="roberta_base_detr_table roberta_base_detr_tableLI"
run_with_synth="roberta_base_with_synthetic_pretraining roberta_ours_with_synthetic_pretraining layoutlmv3_ours_with_synthetic_pretraining"
run="${run_main} ${run_with_detr} ${run_with_synth}"


# The '-user' suffix is here not to overwrite downloaded predictions by mistake
PREDICTIONS_DIR_PREFIX="/app/data/baselines/predictions-user"
mkdir -p ${PREDICTIONS_DIR_PREFIX}

DOCILE_PATH="/app/data/docile"
MODELS_DIR_PREFIX="/app/data/baselines/checkpoints"

# Detections of tables and Line Items. Only used if options --crop_bboxes_filename and/or
# --line_item_bboxes_filename are used as well.
TABLE_TRANSFORMER_PREDICTIONS_DIR="/app/data/baselines/predictions/detr"


function run_inference() {
  cmd=$1
  split=$2
  checkpoint_subdir=$3
  output_dir="${PREDICTIONS_DIR_PREFIX}/$4"
  shift ; shift ; shift ; shift
  extra_params=$@

  mkdir -p $output_dir

  log="${output_dir}/log_inference.txt"

  run_cmd=$(tr '\n' ' ' << EOF
CUDA_VISIBLE_DEVICES=${GPU} python ${cmd}
    --split ${split}
    --docile_path ${DOCILE_PATH}
    --checkpoint "${MODELS_DIR_PREFIX}/${checkpoint_subdir}"
    --output_dir ${output_dir}
    --store_intermediate_results
    --merge_strategy new
    ${extra_params} 2>&1 | tee -a ${log}
EOF
  )
  echo ${run_cmd}
  echo ${run_cmd} >> ${log}
  eval ${run_cmd}
}


CMD_ROBERTA="docile_inference_NER_multilabel.py"
CMD_LAYOUTLMV3="docile_inference_NER_multilabel_layoutLMv3.py"


single_run="roberta_base"
if [[ " ${run} " =~ " ${single_run} " ]]; then
  run_inference ${CMD_ROBERTA} $split roberta_base_65925 $single_run
fi

single_run="roberta_ours"
if [[ " ${run} " =~ " ${single_run} " ]]; then
  run_inference ${CMD_ROBERTA} $split roberta_ours_133352 $single_run
fi

single_run="layoutlmv3_base"
if [[ " ${run} " =~ " ${single_run} " ]]; then
  run_inference ${CMD_LAYOUTLMV3} $split layoutlmv3_base_62100 $single_run
fi

single_run="layoutlmv3_ours"
if [[ " ${run} " =~ " ${single_run} " ]]; then
  run_inference ${CMD_LAYOUTLMV3} $split layoutlmv3_ours_320587 $single_run
fi

# Use table-transformer predictions for table and/or Line Items
single_run="roberta_base_detr_table"
if [[ " ${run} " =~ " ${single_run} " ]]; then
  extra_params="--table_transformer_predictions_dir ${TABLE_TRANSFORMER_PREDICTIONS_DIR} --crop_bboxes_filename docile_td11.json"
  run_inference ${CMD_ROBERTA} $split roberta_base_65925 $single_run ${extra_params}
fi

single_run="roberta_base_detr_tableLI"
if [[ " ${run} " =~ " ${single_run} " ]]; then
  extra_params="--table_transformer_predictions_dir ${TABLE_TRANSFORMER_PREDICTIONS_DIR} --crop_bboxes_filename docile_td11.json --line_item_bboxes_filename docile_td11_tlid13.json"
  run_inference ${CMD_ROBERTA} $split roberta_base_65925 $single_run ${extra_params}
fi

# Models with synthetic pretraining
single_run="roberta_base_with_synthetic_pretraining"
if [[ " ${run} " =~ " ${single_run} " ]]; then
  run_inference ${CMD_ROBERTA} $split roberta_base_with_synthetic_pretraining_125370 $single_run
fi

single_run="roberta_ours_with_synthetic_pretraining"
if [[ " ${run} " =~ " ${single_run} " ]]; then
  run_inference ${CMD_ROBERTA} $split roberta_ours_with_synthetic_pretraining_227640 $single_run
fi

single_run="layoutlmv3_ours_with_synthetic_pretraining"
if [[ " ${run} " =~ " ${single_run} " ]]; then
  run_inference ${CMD_LAYOUTLMV3} $split layoutlmv3_ours_with_synthetic_pretraining_158436 $single_run
fi
