# How to Extract Video Captions

All the captions used in this project are available on [Hugging Face](https://huggingface.co/datasets/lucas-ventura/chapter-llama). However, if you need to extract additional captions for a different subset or try another sampling method, this guide explains our process.

> Note: This is our current implementation, all suggestions are welcome!

## Setup

1. Create a new conda environment:
```bash
conda create python=3.12 -n MiniCPM-V -y
conda activate MiniCPM-V
pip install -r tools/captions/requirements_captioner.txt
```

## Downloading Captions

You can download pre-extracted captions using the `captions.sh` script. This is the recommended way to get started quickly:

```bash
# Download default captions (asr_s10k-2_train_preds+no-asr-10s)
./tools/download/captions.sh

# Or specify a different caption set
./tools/download/captions.sh all
```

Available caption sets include:
- `asr_s10k-2_train_preds+no-asr-10s` (default)
- `asr_s10k-2_train_preds`
- `shot_boundaries`
- `all`

The captions will be downloaded to `dataset/captions/HwwwH_MiniCPM-V-2/`. I recommend creating symlink from your VidChapters directory to the dataset folder using `ln -s path/to/vidchapters/ dataset/`.

## Directory Structure

Captions are stored in `VidChapters/captions/{captioner}/{sampling-method}` where:

- `{captioner}`: Default is `HwwwH_MiniCPM-V-2` (other models can be used)
- `{sampling-method}`: Determines which frames to caption:
  - `all/`: Contains all extracted captions
  - `asr_s10k-2_train_preds/`: Captions using ASR predictions from s10k-2 subset
  - `asr_s10k-2_train_preds+no-asr-10s/`: Uses ASR predictions when available, falls back to 10s intervals
  - `shot_boundaries/`: Uses shot boundary detection
  - And others...

## Extraction Process

### Initial Extraction

Initially, we used `caption_frames.py` to extract the first set of captions. This script starts from 0 and extracts captions based on the `--sampling_methods` parameter. However, it doesn't check for previously extracted captions.

```bash
python tools/captions/caption_frames.py --sampling_methods "10s,60s,shot-detection,asr-preds"
```

### Extracting Missing Captions

For better efficiency, we developed a two-step process:

1. Use `find_missing_captions.ipynb` to:
   - Identify which timestamps need captions
   - Generate a `missing_timestamps/` folder containing the required timestamps

2. Run `caption_frames_timestamp.py` to:
   - Extract only the missing captions
   - Automatically merge new captions into the `all/` subdirectory

```bash
# Example workflow
jupyter notebook tools/captions/find_missing_captions.ipynb
python tools/captions/caption_frames_timestamp.py <shard_id> --subset=<subset_name>
```

### Selecting Captions

Once you have all captions extracted, use `caption_selection.ipynb` to:
- Generate a new folder with your chosen sampling method
- Copy the corresponding captions to this new location

This notebook allows you to experiment with different sampling strategies without re-extracting captions.

## Tips

- The extraction process can be time-consuming - consider using the pre-extracted captions from Hugging Face when possible
- For large datasets, the extraction is parallelized using shards
- Monitor GPU memory usage when running extractions - adjust batch sizes if needed