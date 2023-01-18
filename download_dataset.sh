#!/bin/bash

set -euo pipefail

UNLABELED_CHUNK_MAX=93

print_help() {
  echo "Usage:"
  echo "./download_dataset.sh SECRET_TOKEN DATASET TARGET_DIR [--unzip] [--without-pdfs]"
  echo
  echo "or to only show download urls:"
  echo "./download_dataset.sh SECRET_TOKEN DATASET --show-urls [--without-pdfs]"
  echo
  echo "Script to download the docile dataset. Uses curl to download the files and unzip to unzip"
  echo "them (when --unzip is used). Note: all datasets can be unzipped to the same TARGET_DIR."
  echo
  echo "To get the SECRET_TOKEN, follow the instructions at https://docile.rossum.ai"
  echo
  echo "Available DATASETs: labeled-trainval, synthetic, unlabeled"
  echo "For synthetic and unlabeled datasets you can also download chunks of 10000 documents:"
  echo "    Download stand-alone synthetic chunk (i is in 0..9):"
  echo "        ./download_dataset.sh SECRET_TOKEN synthetic-chunk-\${i} TARGET_DIR [--unzip]"
  echo "    To use unlabeled chunk(s) stand-alone you first need to download annotations"
  echo "    (containing metadata such as page counts and cluster ids):"
  echo "        ./download_dataset.sh SECRET_TOKEN unlabeled-annotations TARGET_DIR [--unzip]"
  echo "    and then the chunk (i is in 00..$UNLABELED_CHUNK_MAX) with:"
  echo "        ./download_dataset.sh SECRET_TOKEN unlabeled-chunk-\${i} TARGET_DIR [--unzip]"
  echo "    or to download only ocrs without pdfs:"
  echo "        ./download_dataset.sh SECRET_TOKEN unlabeled-chunk-ocr-only-\${i} TARGET_DIR [--unzip]"
  echo
  echo "Options:"
  echo "    --unzip           unzip dataset in TARGET_DIR and delete the zip file"
  echo "    --without-pdfs    download without pdfs, can be only used for unlabeled dataset"
}

if [ "$#" -lt 3 ]; then
  echo "You need to provide at least three arguments."
  print_help
  exit 1
fi

token="$1"
dataset="$2"
targetdir="$3"
shift ; shift ; shift

unzip="no"
without_pdfs="no"
while [ "$#" -ge 1 ]; do
  if [[ "$1" == "--without-pdfs" ]]; then
    if [[ "$dataset" != "unlabeled" ]]; then
      echo "--without-pdfs can only be used for unlabeled dataset."
      print_help
      exit 1
    fi
    without_pdfs="yes"
    shift
  elif [[ "$1" == "--unzip" ]]; then
    unzip="yes"
    shift
  else
    echo "Unknown parameter $1."
    print_help
    exit 1
  fi
done

if [[ "$targetdir" != "--show-urls" ]]; then
  mkdir -p "$targetdir"
fi

download_and_unzip() {
  local token="$1"
  local targetdir="$2"
  local unzip="$3"
  local dataset="$4"
  local zipfile="$dataset.zip"
  url="https://docile-dataset-rossum.s3.eu-west-1.amazonaws.com/$token/$zipfile"
  if [[ "$targetdir" == "--show-urls" ]]; then
    echo "$url"
    return 0
  fi
  pushd "$targetdir" > /dev/null
    echo "Downloading $url"
    curl -O "$url"
    zip_size=$(wc -c "$zipfile" | sed -e 's/ *\([0-9]*\) .*/\1/')
    if [ "$zip_size" -lt 100000000 ]; then
      echo "Unexpected size of downloaded $zipfile, perhaps token '$token' or dataset name '$dataset' is wrong"
      exit 1
    fi
    if [[ "$unzip" == "yes" ]]; then
      echo "Unzipping $zipfile"
      unzip -quo "$zipfile"
      rm "$zipfile"
    fi
  popd > /dev/null
}

if [[ "$dataset" == "unlabeled" ]]; then
  download_and_unzip "$token" "$targetdir" "$unzip" "unlabeled-annotations"
  for i in $(seq -f "%02g" 0 "$UNLABELED_CHUNK_MAX"); do
    if [[ "$without_pdfs" == "yes" ]]; then
      download_and_unzip "$token" "$targetdir" "$unzip" "unlabeled-chunk-ocr-only-$i"
    else
      download_and_unzip "$token" "$targetdir" "$unzip" "unlabeled-chunk-$i"
    fi
  done
elif [[ "$dataset" == "baselines" ]]; then
  download_and_unzip "$token" "$targetdir" "$unzip" "baselines-roberta-base"
  download_and_unzip "$token" "$targetdir" "$unzip" "baselines-roberta-ours"
  download_and_unzip "$token" "$targetdir" "$unzip" "baselines-layoutlmv3-ours"
  download_and_unzip "$token" "$targetdir" "$unzip" "baselines-detr"
  download_and_unzip "$token" "$targetdir" "$unzip" "baselines-roberta-pretraining"
  download_and_unzip "$token" "$targetdir" "$unzip" "baselines-layoutlmv3-pretraining"
else
  download_and_unzip "$token" "$targetdir" "$unzip" "$dataset"
fi
