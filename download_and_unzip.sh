#!/bin/bash
if [ "$(uname)" == "Darwin" ]; then
  UNZIP="tar -xf"
else
  UNZIP="unzip -nq"
fi

# Helper function to download and unpack a .zip file.
function download_and_unzip() {
  local BASE_URL=${1}
  local FILENAME=${2}

  if [ ! -f ${FILENAME} ]; then
    echo "Downloading ${FILENAME} to $(pwd)"
    wget "${BASE_URL}/${FILENAME}"
  else
    echo "Skipping download of ${FILENAME}"
  fi
  echo "Unzipping ${FILENAME}"
  ${UNZIP} ${FILENAME}
}

# Download the captions.
URL="https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition/download"
TRAIN="train.zip"
TEST="test.zip"

download_and_unzip ${URL} ${TRAIN}
download_and_unzip ${URL} ${TEST}
