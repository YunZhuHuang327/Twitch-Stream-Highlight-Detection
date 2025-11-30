#!/bin/bash

# Color definitions
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Parse arguments
OVERWRITE=false
SUBSET=""

for arg in "$@"; do
    if [ "$arg" = "--overwrite" ]; then
        OVERWRITE=true
    else
        SUBSET=$arg
    fi
done

# Check if subset argument is provided
if [ -z "$SUBSET" ]; then
    echo -e "${RED}Error: Please provide a subset name${NC}"
    echo "Usage: $0 <subset> [--overwrite]"
    echo "Available subsets:"
    echo "full"
    echo "l100_val l_test m100_val m_test s100_val s10k-2_train s_test sml10k_train sml1k_train sml300_val test eval"
    echo ""
    echo "Options:"
    echo "  --overwrite    Overwrite existing files"
    exit 1
fi

# Base URL for the files
BASE_URL="https://huggingface.co/datasets/lucas-ventura/chapter-llama/resolve/main"

# Create target directory structure
TARGET_DIR="dataset/docs"
if [ "$SUBSET" != "full" ]; then
    TARGET_DIR="$TARGET_DIR/subset_data"
    mkdir -p "$TARGET_DIR"
    mkdir -p "$TARGET_DIR/asrs"
    mkdir -p "$TARGET_DIR/chapters"
else
    mkdir -p "$TARGET_DIR"
fi

# Function to download a file
download_file() {
    local url=$1
    local output=$2
    local filename=$(basename $output)
    
    # Check if file exists
    if [ -f "$output" ]; then
        if [ "$OVERWRITE" = true ]; then
            echo -e "${YELLOW}File ${filename} exists. Overwriting...${NC}"
        else
            echo -e "${BLUE}Skipping ${filename} (already exists). Use --overwrite to force download${NC}"
            return 0
        fi
    fi
    
    echo -e "Downloading ${BLUE}${filename}${NC}..."
    
    # Download with status code checking
    local http_code=$(curl -s -w "%{http_code}" -L "${url}\?download\=true" -o "$output")
    
    # Check for specific error cases
    if grep -q "Entry not found" "$output" 2>/dev/null; then
        echo -e "${RED}Error: File ${filename} not found on the server${NC}"
        rm -f "$output"  # Clean up the error message file
        return 1
    fi
    
    if [ "$http_code" != "200" ]; then
        echo -e "${RED}Error: Failed to download ${filename} (HTTP status: ${http_code})${NC}"
        rm -f "$output"  # Clean up any partial download
        return 1
    fi
    
    # Verify the downloaded file is not empty
    if [ ! -s "$output" ]; then
        echo -e "${RED}Error: Downloaded file ${filename} is empty${NC}"
        rm -f "$output"
        return 1
    fi
    
    return 0
}

# Download files based on subset
echo -e "Downloading files for: ${BLUE}${SUBSET}${NC}"

if [ "$SUBSET" = "full" ]; then
    # Download full documentation files
    download_file "${BASE_URL}/docs/asrs.json" "${TARGET_DIR}/asrs.json"
    if [ $? -ne 0 ]; then exit 1; fi
    
    download_file "${BASE_URL}/docs/chapters.json" "${TARGET_DIR}/chapters.json"
    if [ $? -ne 0 ]; then exit 1; fi
else
    # Download subset-specific files
    download_file "${BASE_URL}/docs/subset_data/${SUBSET}.json" "${TARGET_DIR}/${SUBSET}.json"
    if [ $? -ne 0 ]; then exit 1; fi
    
    download_file "${BASE_URL}/docs/subset_data/asrs/asrs_${SUBSET}.json" "${TARGET_DIR}/asrs/asrs_${SUBSET}.json"
    if [ $? -ne 0 ]; then exit 1; fi
    
    download_file "${BASE_URL}/docs/subset_data/chapters/chapters_${SUBSET}.json" "${TARGET_DIR}/chapters/chapters_${SUBSET}.json"
    if [ $? -ne 0 ]; then exit 1; fi
fi

echo -e "${GREEN}Successfully downloaded all files for ${SUBSET}${NC}"