import os
from pathlib import Path

import hydra
from hydra.utils import instantiate
from lightning.fabric import Fabric
from lightning.fabric.utilities.seed import seed_everything
from omegaconf import DictConfig
from torch.cuda import empty_cache

from src.utils import RankedLogger, extras, task_wrapper

log = RankedLogger("train", rank_zero_only=True)
empty_cache()


@task_wrapper
def train(cfg: DictConfig):
    """Trains the model.

    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.

    :param cfg: A DictConfig configuration composed by Hydra.
    """
    # Check if finetuned model is already present
    model_path = Path(cfg.model.config_train.output_dir) / "adapter_model.safetensors"
    if model_path.exists():
        log.warning(f"Model checkpoint already exists at {model_path}")
        return

    # Check if pretrained model exists locally. If not, assume it's a HF identifier and continue
    model_path = Path(cfg.model.config_train.model_name)
    if not model_path.exists():
        log.info(
            f"Model path {model_path} does not exist locally. Will attempt to load from Hugging Face using the identifier '{cfg.model.config_train.model_name}'"
        )

    log.info(f"Output dir: {cfg.paths.output_dir}")

    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        seed_everything(cfg.seed, workers=True)

    fabric = Fabric(accelerator="gpu", strategy="auto", devices=1, num_nodes=1)
    fabric.launch()
    os.environ["LOCAL_RANK"] = str(fabric.local_rank)
    os.environ["RANK"] = str(fabric.global_rank)

    # Most likely src.data.vidchapters.VidChaptersData
    log.info(f"Instantiating train_data <{cfg.data._target_}>")
    train_data = instantiate(cfg.data)

    # Most likely src.models.llama_finetune.Trainer
    log.info(f"Instantiating model <{cfg.model.trainer._target_}>")
    trainer = instantiate(cfg.model.trainer)

    log.info("Starting training!")
    trainer.fit(model_config=cfg.model.config_train, datamodule=train_data)

    # Wait for all GPUs to finish
    fabric.barrier()

    log.info("Done training!")
    log.info(f"Output model dir: {cfg.model.config_train.output_dir}")


@hydra.main(version_base="1.3", config_path="configs", config_name="train.yaml")
def main(cfg: DictConfig):
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    extras(cfg)

    train(cfg)


if __name__ == "__main__":
    main()
