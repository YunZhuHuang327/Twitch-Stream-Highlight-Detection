from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

REPO_ID = "lucas-ventura/chapter-llama"

# Dictionary mapping short identifiers to full model paths
MODEL_PATHS = {
    "asr-1k": "outputs/chapterize/Meta-Llama-3.1-8B-Instruct/asr/default/sml1k_train/default/model_checkpoints/",
    "asr-10k": "outputs/chapterize/Meta-Llama-3.1-8B-Instruct/asr/default/s10k-2_train/default/model_checkpoints/",
    "captions_asr-1k": "outputs/chapterize/Meta-Llama-3.1-8B-Instruct/captions_asr/asr_s10k-2_train_preds+no-asr-10s/sml1k_train/default/model_checkpoints/",
    "captions_asr-10k": "outputs/chapterize/Meta-Llama-3.1-8B-Instruct/captions_asr/asr_s10k-2_train_preds+no-asr-10s/sml10k_train/default/model_checkpoints/",
}

FILES = ["adapter_model.safetensors", "adapter_config.json"]


def download_model(model_id_or_path, overwrite=False, local_dir=None):
    # Get filename from aliases or use the provided path
    model_path = MODEL_PATHS.get(model_id_or_path, model_id_or_path)

    for file in FILES:
        try:
            # Use forward slashes for HuggingFace URLs (works on all platforms)
            file_path = model_path.rstrip('/') + '/' + file
            cache_path = hf_hub_download(
                repo_id=REPO_ID,
                filename=file_path,
                force_download=overwrite,
                local_dir=local_dir,
            )

            if not overwrite:
                print(f"File {file} found in cache at: {cache_path}")
            else:
                print(f"File {file} downloaded to: {cache_path}")

        except Exception as e:
            print(f"Error downloading {file}: {e}")
            return None

    print("All files loaded successfully")
    return str(Path(cache_path).parent)


def download_base_model(
    repo_id="meta-llama/Meta-Llama-3.1-8B-Instruct",
    local_dir="./checkpoints/",
):
    """
    Downloads the base model from Hugging Face Hub.

    Args:
        repo_id (str): The repository ID on Hugging Face
        local_dir (str): Directory to save the model to

    Returns:
        str: Path to the downloaded model directory
    """
    try:
        print(f"Downloading {repo_id} to {local_dir}...")
        model_path = snapshot_download(repo_id=repo_id, local_dir=local_dir)
        print(f"Model downloaded successfully to: {model_path}")
        return model_path
    except Exception as e:
        print(f"Error downloading model {repo_id}: {e}")
        print(
            f"\nYou are downloading `{repo_id}` to `{local_dir}` but failed. "
            f"Please accept the agreement and obtain access at https://huggingface.co/{repo_id}. "
            f"Then, use `huggingface-cli login` and your access tokens at https://huggingface.co/settings/tokens to authenticate. "
            f"After that, run the code again."
        )
        return None


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Download models from Hugging Face Hub"
    )
    parser.add_argument(
        "model_id", type=str, help="ID or full path of the model to download"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Force re-download even if the model exists in cache",
    )
    parser.add_argument(
        "--local_dir",
        type=str,
        default=None,
        help="Download to local directory instead of cache",
    )
    args = parser.parse_args()

    model_dir = download_model(
        args.model_id, overwrite=args.overwrite, local_dir=args.local_dir
    )
    print(f"Model directory: {model_dir}")
