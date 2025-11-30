from pathlib import Path

import hydra
from hydra.utils import instantiate
from lightning.fabric import Fabric
from lightning.fabric.utilities.seed import seed_everything
from omegaconf import DictConfig, OmegaConf
from torch.cuda import empty_cache

from src.utils import RankedLogger, extras, task_wrapper

log = RankedLogger(__name__, rank_zero_only=True)
empty_cache()


@task_wrapper
def test(cfg: DictConfig):
    """Trains the model. Can additionally evaluate on a testset if provided.

    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.

    :param cfg: A DictConfig configuration composed by Hydra.
    :return: A tuple with metrics and dict with all instantiated objects.
    """
    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        seed_everything(cfg.seed, workers=True)

    fabric = Fabric(accelerator="gpu", strategy="ddp", devices="auto", num_nodes=1)
    fabric.launch()

    # Most likely src.models.llama_inference.LlamaInference
    inference = instantiate(cfg.model.config_inference)
    if inference is None:
        return

    test_data_cfg = OmegaConf.to_container(cfg.test.data, resolve=True)

    if isinstance(test_data_cfg["subset"], str):
        test_data_cfg["subset"] = [test_data_cfg["subset"]]

    for subset in test_data_cfg["subset"]:
        # Update the subset for testing to ensure we're not using the training subset
        test_data_cfg["prompter"]["chapters"]["subset"] = subset
        log.info(f"Testing on {subset}")
        log.info(f"Output dir: {cfg.test.save_dir}")

        # Most likely src.data.vidchapters.VidChaptersData
        test_data = instantiate(test_data_cfg)
        test_dataloader = fabric.setup_dataloaders(test_data.test_dataloader())

        # Most likely src.test.vidchapters.VidChaptersTester
        test = instantiate(cfg.test)
        test(
            inference=inference,
            test_dataloader=test_dataloader,
        )

        # Wait for all GPUs to finish
        fabric.barrier()

    log.info("Done testing!")


@hydra.main(version_base="1.3", config_path="configs", config_name="test.yaml")
def main(cfg: DictConfig):
    """Main entry point for training.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Optional[float] with optimized metric value.
    """
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    extras(cfg)

    # test the model
    test(cfg)


if __name__ == "__main__":
    main()
