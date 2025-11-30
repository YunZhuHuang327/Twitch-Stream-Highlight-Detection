"""
Download Llama 3.1 8B Instruct base model from HuggingFace

This will download the model to the HuggingFace cache, so chapter_from_asr_english.py can use it.
"""

import os
from huggingface_hub import snapshot_download

def download_llama_base():
    """Download Llama 3.1 8B Instruct base model"""

    repo_id = "meta-llama/Llama-3.1-8B-Instruct"

    print("="*60)
    print("Downloading Llama 3.1 8B Instruct Base Model")
    print("="*60)
    print(f"Repository: {repo_id}")
    print(f"\nThis will download ~16GB of model files.")
    print(f"Files will be cached in: {os.path.expanduser('~/.cache/huggingface/hub')}")
    print("\nIMPORTANT:")
    print("1. You need to accept the Llama license at:")
    print("   https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct")
    print("2. You need to login with your HuggingFace token:")
    print("   huggingface-cli login")
    print("="*60)

    response = input("\nHave you completed steps 1 and 2? (y/n): ").lower()
    if response != 'y':
        print("\nPlease complete the steps above first:")
        print("1. Visit https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct")
        print("2. Click 'Agree and access repository'")
        print("3. Run: huggingface-cli login")
        print("4. Enter your access token from https://huggingface.co/settings/tokens")
        return None

    print("\nDownloading model... This may take 10-30 minutes depending on your internet speed.")

    try:
        model_path = snapshot_download(
            repo_id=repo_id,
            cache_dir=None,  # Use default cache
        )

        print("\n" + "="*60)
        print("SUCCESS!")
        print("="*60)
        print(f"Model downloaded to: {model_path}")
        print("\nYou can now run:")
        print("  .\\run_chapter_english.bat")
        print("\nOr:")
        print("  python chapter_from_asr_english.py --asr_file \"dataset/highlights/123/asr.txt\" --video_title \"TwitchCon W/ AGENT00\" --output_dir \"outputs/chapters/123\"")
        print("="*60)

        return model_path

    except Exception as e:
        print("\n" + "="*60)
        print("ERROR!")
        print("="*60)
        print(f"Failed to download model: {e}")
        print("\nCommon issues:")
        print("1. Not logged in - run: huggingface-cli login")
        print("2. Haven't accepted license - visit: https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct")
        print("3. Network issues - try again later")
        print("="*60)
        return None


if __name__ == "__main__":
    download_llama_base()
