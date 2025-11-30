from pathlib import Path

import hydra
from hydra.core.global_hydra import GlobalHydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf
from tqdm import tqdm


def load_config(exp_dir, config_name="config", hydra_dir=".hydra_train"):
    GlobalHydra.instance().clear()

    hydra_path = exp_dir / ".." / hydra_dir
    if not hydra_path.exists():
        hydra_path = exp_dir / hydra_dir
    if not hydra_path.exists():
        if hydra_dir == ".hydra_train":
            return load_config(exp_dir, config_name, hydra_dir=".hydra_test")
        else:
            raise FileNotFoundError(f"Hydra config not found in {hydra_path}")

    hydra.initialize(config_path=str(hydra_path), version_base="1.3")
    cfg = hydra.compose(config_name=config_name, return_hydra_config=True)
    HydraConfig().cfg = cfg
    OmegaConf.resolve(cfg)

    return cfg


def find_matching_paths(
    config_var, config_fix=None, base_dir=Path("../../"), verbose=False
):
    config_fix = config_fix or {}
    config_default = {
        "task_name": "chapterize",
        "model_name": "Meta-Llama-3.1-8B-Instruct",
        "prompt": "*",
        "data_flags": "*",
        "subset": "*",
        "model_flags": "default",
        "test": "test",
    }

    log_dir = base_dir / "outputs/"

    matching_paths = {}
    for key, config in config_var.items():
        # Combine config_fix and config
        full_config = {**config_default, **config_fix, **config}

        # Construct the path pattern
        path_pattern = (
            f"{full_config['task_name']}/"
            f"{full_config['model_name']}/"
            f"{full_config['prompt']}/"
            f"{full_config['data_flags']}/"
            f"{full_config['subset']}/"
            f"{full_config['model_flags']}/"
            f"{full_config['test']}/"
        )

        # Find all matching paths
        found_paths = list(log_dir.glob(path_pattern))
        if len(found_paths) == 0:
            print(f"No paths found for {path_pattern}")
            continue

        matching_paths[key] = found_paths

        if not found_paths and verbose:
            print(f"No matching paths found for {key} with pattern {path_pattern}")

    return matching_paths


def find_common_path(paths):
    # Convert the list of paths to Path objects if not already
    paths = [Path(p) for p in paths]

    # Use the first path as the starting point
    common_path = paths[0]

    # Iterate through the rest of the paths and find the common part
    for path in paths[1:]:
        # Get the common part by using parts of the paths and stopping at the mismatch
        common_path = Path(*common_path.parts[: len(Path(*common_path.parts).parts)])
        for i, (common_part, current_part) in enumerate(
            zip(common_path.parts, path.parts)
        ):
            if common_part != current_part:
                common_path = Path(*common_path.parts[:i])
                break

    return common_path


def format_number_with_z(num, num_digits=2):
    """Format number with 1 decimal point and add z if fewer than num_digits before decimal"""
    num_str = f"{num:.1f}"
    digits_before = len(str(int(num)))
    if digits_before < num_digits:
        return r"\z " * (num_digits - digits_before) + num_str
    return num_str


def tokens_per_min(exp_dir, subset):
    import numpy as np
    from hydra.utils import instantiate

    from src.models.utils_tokenizer import Tokenizer

    path = exp_dir
    while path.name != "outputs":
        path = path.parent
    base_dir = path.parent
    vidc_dir = base_dir / "dataset"

    cfg = load_config(exp_dir)

    cfg["data"]["prompter"]["chapters"]["subset"] = subset

    cfg["data"]["prompter"]["chapters"]["vidc_dir"] = vidc_dir
    if "captions_dir" in cfg["data"]["prompter"]["chapters"]:
        cfg["data"]["prompter"]["chapters"]["captions_dir"] = cfg["data"]["prompter"][
            "chapters"
        ]["captions_dir"].replace("./dataset/", "../../dataset/")

    try:
        data = instantiate(cfg["data"])
    except Exception as e:
        print(f"Error instantiating {subset} from {exp_dir}: {e}")
        raise e

    tokenizer = Tokenizer(base_dir=base_dir)

    tokens_per_min = []
    for vid_data in tqdm(data):
        prompt = vid_data["transcript"]
        vid_duration = data.chapters.hms_to_sec(vid_data["vid_duration"])

        vid_tokens = tokenizer.n_tokens(prompt)
        tokens_per_min.append(vid_tokens / vid_duration * 60)

    return round(np.mean(tokens_per_min), 2)
