<div align="center">

# Chapter-Llama
## Efficient Chaptering in Hour-Long Videos with LLMs

<a href="http://lucasventura.com/"><strong>Lucas Ventura</strong></a>
·
<a href="https://antoyang.github.io/"><strong>Antoine Yang</strong></a>
·
<a href="https://www.di.ens.fr/willow/people_webpages/cordelia/"><strong>Cordelia Schmid</strong></a>
·
<a href="https://imagine.enpc.fr/~varolg"><strong>G&#252;l Varol</strong></a>

![CVPR 2025](https://img.shields.io/badge/CVPR-2025-1B427D)

[![arXiv](https://img.shields.io/badge/arXiv-Chapter--Llama-9065CA.svg?logo=arXiv)](https://arxiv.org/abs/2504.00072)
[![Project Page](https://img.shields.io/badge/Project-Page-3E7DFF?logo=data:image/svg%2bxml;base64,PCFET0NUWVBFIHN2ZyBQVUJMSUMgIi0vL1czQy8vRFREIFNWRyAxLjEvL0VOIiAiaHR0cDovL3d3dy53My5vcmcvR3JhcGhpY3MvU1ZHLzEuMS9EVEQvc3ZnMTEuZHRkIj4KDTwhLS0gVXBsb2FkZWQgdG86IFNWRyBSZXBvLCB3d3cuc3ZncmVwby5jb20sIFRyYW5zZm9ybWVkIGJ5OiBTVkcgUmVwbyBNaXhlciBUb29scyAtLT4KPHN2ZyB3aWR0aD0iODAwcHgiIGhlaWdodD0iODAwcHgiIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiBzdHJva2U9IiMwMDAwMDAiPgoNPGcgaWQ9IlNWR1JlcG9fYmdDYXJyaWVyIiBzdHJva2Utd2lkdGg9IjAiLz4KDTxnIGlkPSJTVkdSZXBvX3RyYWNlckNhcnJpZXIiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIgc3Ryb2tlLWxpbmVqb2luPSJyb3VuZCIvPgoNPGcgaWQ9IlNWR1JlcG9faWNvbkNhcnJpZXIiPiA8cGF0aCBkPSJNMyA2QzMgNC4zNDMxNSA0LjM0MzE1IDMgNiAzSDE0QzE1LjY1NjkgMyAxNyA0LjM0MzE1IDE3IDZWMTRDMTcgMTUuNjU2OSAxNS42NTY5IDE3IDE0IDE3SDZDNC4zNDMxNSAxNyAzIDE1LjY1NjkgMyAxNFY2WiIgc3Ryb2tlPSIjODFhOWQwIiBzdHJva2Utd2lkdGg9IjIiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIgc3Ryb2tlLWxpbmVqb2luPSJyb3VuZCIvPiA8cGF0aCBkPSJNMjEgN1YxOEMyMSAxOS42NTY5IDE5LjY1NjkgMjEgMTggMjFINyIgc3Ryb2tlPSIjODFhOWQwIiBzdHJva2Utd2lkdGg9IjIiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIgc3Ryb2tlLWxpbmVqb2luPSJyb3VuZCIvPiA8cGF0aCBkPSJNOSAxMlY4TDEyLjE0MjkgMTBMOSAxMloiIGZpbGw9IiM4MWE5ZDAiIHN0cm9rZT0iIzgxYTlkMCIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiLz4gPC9nPgoNPC9zdmc+)](https://imagine.enpc.fr/~lucas.ventura/chapter-llama/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)]()
[![GitHub Stars](https://img.shields.io/github/stars/lucas-ventura/chapter-llama?style=social)](https://github.com/lucas-ventura/chapter-llama)


</div>

<div align="center">
<img src="https://imagine.enpc.fr/~lucas.ventura/chapter-llama/images/teaser.png" alt="Chapter-Llama teaser" width="500">
</div>

<div align="justify">

> We address the task of video chaptering, i.e., partitioning a long video timeline into semantic units and generating corresponding chapter titles. While relatively underexplored, automatic chaptering has the potential to enable efficient navigation and content retrieval in long-form videos. In this paper, we achieve strong chaptering performance on hour-long videos by efficiently addressing the problem in the text domain with our 'Chapter-Llama' framework. Specifically, we leverage a pretrained large language model (LLM) with large context window, and feed as input (i) speech transcripts and (ii) captions describing video frames, along with their respective timestamps. Given the inefficiency of exhaustively captioning all frames, we propose a lightweight speech-guided frame selection strategy based on speech transcript content, and experimentally demonstrate remarkable advantages. We train the LLM to output timestamps for the chapter boundaries, as well as free-form chapter titles. This simple yet powerful approach scales to processing one-hour long videos in a single forward pass. Our results demonstrate substantial improvements (e.g., 45.3 vs 26.7 F1 score) over the state of the art on the recent VidChapters-7M benchmark. To promote further research, we release our code and models.

</div>

## Description
This repository contains the code for the paper ["Chapter-Llama: Efficient Chaptering in Hour-Long Videos with LLMs"](https://arxiv.org/abs/2504.00072) (CVPR 2025).

Please visit our [webpage](http://imagine.enpc.fr/~lucas.ventura/chapter-llama/) for more details.


<details><summary><strong>Project Structure</strong></summary>
&emsp; 

```
📦 chapter-llama/
├── 📂 configs/            # Hydra configuration files
│   ├── 📂 data/           # Data loading configurations
│   │   ├── asr.yaml       # ASR-only data loading
│   │   ├── captions.yaml  # Captions-only data loading
│   │   ├── ...
│   │   └── captions_asr.yaml # Combined captions and ASR
│   ├── 📂 experiment/     # Experiment-specific configs
│   ├── 📂 model/          # Model architectures and parameters
│   ├── 📄 test.yaml       # Test configuration
│   └── 📄 train.yaml      # Main training configuration
├── 📂 src/
│   ├── 📂 data/
│   ├── 📂 models/
│   ├── 📂 test/
│   └── 📂 utils/
├── 📂 tools/
│   ├── 📂 captions/       # Caption extraction
│   ├── 📂 download/       # Download data (captions, docs, models, ids)
│   ├── 📂 extract/        # Extract embeddings
│   ├── 📂 results/        # Visualize results
│   ├── 📂 shot_detection/ # Shot detection
│   └── 📂 slurm/          # SLURM job submission
├── 🗃️ dataset/            # symlink to VidChapters dataset
├── 🗄️ outputs/            # output directory
├── 📄 inference.py        # Inference script
├── 📄 test.py             # Testing script
└── 📄 train.py            # Main training script
```
</details>


## Installation 👷

<details><summary><strong>Create environment</strong></summary>
&emsp; 

```bash
conda create python=3.12 --name chapter-llama -y
conda activate chapter-llama
```

To install the necessary packages for training and testing, you can use the provided requirements.txt file:
```bash
python -m pip install -r requirements.txt
```
or 
```bash
python -m pip install -e .
```

The code was tested on Python 3.12.9 and PyTorch 2.6.0

For inference on videos that are not in the VidChapters-7M dataset, 
you will need to install the following dependencies to extract ASR and captions from the video (not required for training):
```bash
python -m pip install -e ".[inference]"
```

</details>

<details><summary><strong>Download models</strong></summary>
&emsp; 

The `Llama-3.1-8B-Instruct` model will be downloaded automatically from Hugging Face, 
make sure you agree to the license terms [here](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct).
If you already have it downloaded, please check the [`llama3.1_8B.yaml`](configs/model/llama3.1_8B.yaml) config file to specify the checkpoint path.

We provide 3 LoRA parameter sets for Llama-3.1-8B-Instruct:
- `asr-10k`: Model trained with ASR from 10k videos of the VidChapters-7M dataset. Used for our Speech-based frame selector.
- `captions_asr-10k`: Model trained with Captions+ASR from 10k videos of the VidChapters-7M dataset. Used for most of our experiments.
- `captions_asr-1k`: Model trained with Captions+ASR from 1k videos of the VidChapters-7M dataset. Used for the full test set.

To download the LoRA parameter sets, run:
```bash
python tools/download/models.py "asr-1k" --local_dir "."
python tools/download/models.py "asr-10k" --local_dir "."
python tools/download/models.py "captions_asr-1k" --local_dir "."
python tools/download/models.py "captions_asr-10k" --local_dir "."
```

</details>


<details><summary><strong>Download captions</strong></summary>
&emsp; 

First, create a directory to store the data and create a symlink to it from the VidChapters directory:
```bash
mkdir path/to/VidChapters/
ln -s path/to/VidChapters/ dataset/
```

The `dataset/captions/` directory contains the video captions organized by the captioning model used (`HwwwH_MiniCPM-V-2`) and the sampling method. 
To download the captions for our sampling method (`asr_s10k-2_train_preds+no-asr-10s`), run:
```bash
bash tools/download/captions.sh asr_s10k-2_train_preds+no-asr-10s
```
This method uses predictions from an ASR model trained on the `s10k-2` subset when ASR is available for the video, 
and falls back to sampling frames every 10 seconds when ASR is not available.
The other sampling methods available are:
```bash
bash tools/download/captions.sh 10s
bash tools/download/captions.sh 100f
bash tools/download/captions.sh shot_boundaries
```
Please refer to the [how_to_extract_captions.md](tools/captions/how_to_extract_captions.md) documentation for more details.

</details>

<details><summary><strong>Download docs</strong></summary>
&emsp; 

The `dataset/docs/` directory contains the ASR and chapter data for the VidChapters-7M dataset. We provide:
1. **Complete dataset**: Contains 817k videos with approximately 20 GB of ASR data.
2. **Specific subsets** used in our paper experiments (recommended for most users as it is much faster to load and download).

To download a specific subset's data (which includes video ids, ASR and chapter information), run:
```bash
bash tools/download/docs.sh subset_name
```

Where `subset_name` can be `full` for the complete dataset or one of the following:

Training sets:
- `sml1k_train`: Training set with 1k videos (**s**hort+**m**edium+**l**ong), used for ablation studies.
- `sml10k_train`: Training set with 10k videos (**s**hort+**m**edium+**l**ong), used for Table 1 and the data scaling experiment (Figure 4).
- `s10k-2_train`: Training set with 10k videos (**s**hort), used for our Speech-based frame selector.

Validation sets:
- `sml300_val`: Validation set with 300 videos (**s**hort+**m**edium+**l**ong)
- `s100_val`: Validation set with 100 videos (**s**hort)
- `m100_val`: Validation set with 100 videos (**m**edium)
- `l100_val`: Validation set with 100 videos (**l**ong)

Test sets:
- `s_test`: **S**hort videos (<15 min) from the test set
- `m_test`: **M**edium videos (15-30 min) from the test set
- `l_test`: **L**ong videos (30-60 min) from the test set
- `test`: All videos from the test set
- `eval`: All videos from the test + validation sets

If you want to train/test with a different subset of videos,
you can generate a file at `dataset/docs/subset_data/subset_name.json` 
with the video ids you want to use.
The ASR and chapter data will be created automatically when calling the `Chapter` class.

</details>


<details><summary><strong>Download videos</strong></summary>
&emsp;

**Note:** 
Downloading videos is not necessary for training or testing since we provide the extracted captions and ASR data via Hugging Face (see commands above). 
This step is only required if you want to process new videos from different subsets 
or extract additional captions.

To download videos, install the [`yt-dlp`](https://pypi.org/project/yt-dlp/) library and run:
```bash
python tools/download/videos.py dataset/docs/subset_data/subset_name.json
```

The videos will be downloaded to `dataset/videos/`.

If you encounter any issues with `yt-dlp`, you can find the video subsets `sml1k_train`, `sml300_val`, and `test` on Hugging Face.
Access them here: [Hugging Face Datasets - lucas-ventura/chapter-llama](https://huggingface.co/datasets/lucas-ventura/chapter-llama/tree/main/videos).


</details>




<details><summary><strong>Dataset structure</strong></summary>
&emsp; 
Here's how the dataset folder should be structured:

```
dataset/
├── captions/
│   └── HwwwH_MiniCPM-V-2/
│       ├── 100f/
│       ├── ...
│       └── asr_s10k-2_train_preds+no-asr-10s/  # You only need this one
├── docs/
│   ├── asrs.json                               # Optional, ASR for the full dataset
│   ├── chapters.json                           # Optional, Chapter data for the full dataset
│   └── subset_data/
│       ├── sml1k_train.json                    # Video ids for our training subset
│       ├── asrs/
│       │   └── asrs_sml1k_train.json           # ASR data for our training subset
│       ├── chapters/
│       │   └── chapters_sml1k_train.json       # Chapter data for our training subset
│       └── ...
├── videos/                                     # Optional, for testing on new videos
└── embs/                                       # Optional, for embedding experiments
```

</details>

## Usage 💻

<details><summary><strong>Training and testing</strong></summary>
&emsp; 

<!-- TODO:export PYTHONPATH=$PYTHONPATH:$(pwd) -->

The command to launch a training experiment is the following:
```bash
python train.py [OPTIONS]
```   
or to run both train.py and test.py with the same options:
```bash
bash train.sh [OPTIONS]
```

For only testing, run:
```bash
python test.py [OPTIONS]
```

Note: You might need to run the test script with a single GPU (`CUDA_VISIBLE_DEVICES=0`). 

</details>


<details><summary><strong>Configuration</strong></summary>
&emsp; 

The project uses Hydra for configuration management. Key configuration options:

- `data`: asr, captions, captions_asr, captions_asr_given_times, captions_asr_given_titles, captions_asr_window
- `subset_train`: Training dataset subset (default: "s1k_train")
- `paths`: To change default paths, create a `default.yaml` file in `configs/local/` and modify it as in `configs/local/example.yaml`
- `model`: `llama3.1_8B` (default), `zero-shot`, `llama3.2_3B`, etc.

For example, to run training with the `sml1k_train` subset with ASR only, run:
```bash
bash train.sh data=asr subset_train=sml1k_train subset_test=sml300_val
```


</details>


<details><summary><strong>Results</strong></summary>
&emsp; 

To get results from a single test experiment, run:
```bash
python tools/results/evaluate_results.py path/to/experiment/test_dir --subset subset_name
```

For example:
```bash
python tools/results/evaluate_results.py outputs/chapterize/Meta-Llama-3.1-8B-Instruct/captions_asr/asr_s10k-2_train_preds+no-asr-10s/sml1k_train/default/test/
```

Additionally, you can use the `tools/results/evaluate_results.ipynb` notebook to compare results from different video chapter generation experiments.

</details>

## Quick Start ⚡

<details><summary><strong>Single Video Chaptering 📹</strong></summary>
&emsp; 

If you just want to generate chapters for a single video:

```bash
# Clone the repository
git clone https://github.com/lucas-ventura/chapter-llama.git
cd chapter-llama

# Create a conda environment (optional but recommended)
conda create python=3.12 --name chapter-llama -y
conda activate chapter-llama

# Install the package with inference dependencies
python -m pip install -e ".[inference]"

# Run the chaptering command
python inference.py /path/to/your/video.mp4
```

Currently, the command only uses the audio (via ASR extraction) to generate chapters.
Support for automatic visual caption extraction will be added soon.

Chapters and the full output text will be saved in `outputs/inference/<video_name>/`.

</details>


<details><summary><strong>Interactive Demo 🎮</strong></summary>
&emsp; 

We also provide an interactive demo where you can upload videos and visualize the generated chapters:

```bash
# Clone the repository
git clone https://github.com/lucas-ventura/chapter-llama.git
cd chapter-llama

# Install demo dependencies
python -m pip install -e ".[demo]"

# Launch the demo
python demo.py
```

The demo will start a local web server that you can access in your browser.
You can upload videos, generate chapters, and see them visualized on the video timeline.

</details>

## Citation 📝
If you use this code in your work, please cite our [paper](https://arxiv.org/abs/2504.00072):

```bibtex
@InProceedings{ventura25chapter,
    title     = {{Chapter-Llama}: Efficient Chaptering in Hour-Long Videos with {LLM}s},
    author    = {Lucas Ventura and Antoine Yang and Cordelia Schmid and G{"u}l Varol},
    booktitle = {CVPR},
    year      = {2025}
}
```

## Acknowledgements
Based on [llama-cookbook](https://github.com/meta-llama/llama-cookbook) and [lightning-hydra-template](https://github.com/ashleve/lightning-hydra-template/tree/main).



## License :books:
This code is distributed under an [MIT License](LICENSE).

Our models are trained using the VidChapters-7M dataset, which has its own license that must be followed. Additionally, this project depends on several third-party libraries and resources, each with their own licenses: [PyTorch](https://github.com/pytorch/pytorch/blob/master/LICENSE), [Hugging Face Transformers](https://github.com/huggingface/transformers/blob/main/LICENSE), [Hydra](https://github.com/facebookresearch/hydra/blob/main/LICENSE), [Lightning](https://github.com/Lightning-AI/lightning/blob/master/LICENSE), [Llama models](https://github.com/meta-llama/llama/blob/main/LICENSE).

