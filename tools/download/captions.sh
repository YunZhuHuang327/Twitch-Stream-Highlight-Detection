#!/bin/bash

# Color definitions
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check if captions argument is provided
if [ -z "$1" ]; then
    echo -e "Using default captions: ${BLUE}asr_s10k-2_train_preds+no-asr-10s${NC}"
    CAPTIONS="asr_s10k-2_train_preds+no-asr-10s"
else
    CAPTIONS=$1
fi

# Replace + with %2B in URL encoding
if [[ "$CAPTIONS" == *"+"* ]]; then
    URL_CAPTIONS="${CAPTIONS//+/%2B}"
else
    URL_CAPTIONS="$CAPTIONS"
fi

TARGET_DIR="dataset/captions/HwwwH_MiniCPM-V-2"
ZIP_URL="https://huggingface.co/datasets/lucas-ventura/chapter-llama/resolve/main/${URL_CAPTIONS}.zip?download=true"
# Create target directory if it doesn't exist
mkdir -p "$TARGET_DIR"

# Download the zip file
echo -e "Downloading ${BLUE}${CAPTIONS}.zip${NC}..."
curl -L "$ZIP_URL" -o "${TARGET_DIR}/${CAPTIONS}.zip"

# Check if download was successful
if [ $? -ne 0 ]; then
    echo -e "${RED}Error: Failed to download the zip file${NC}"
    exit 1
fi

# Extract the zip file
echo -e "Extracting ${BLUE}${CAPTIONS}.zip${NC}..."
unzip -q -o "${TARGET_DIR}/${CAPTIONS}.zip" -d "$TARGET_DIR"

# Check if extraction was successful
if [ $? -ne 0 ]; then
    echo -e "${RED}Error: Failed to extract the zip file${NC}"
    exit 1
fi

# Remove the zip file after successful extraction
rm "${TARGET_DIR}/${CAPTIONS}.zip"

echo -e "${GREEN}Successfully downloaded and extracted ${CAPTIONS} to ${TARGET_DIR}${NC}"
