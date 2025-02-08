import rootutils

rootutils.setup_root(__file__, ".project-root", pythonpath=True)

import warnings

warnings.filterwarnings("ignore")

import logging
import os

import hydra
import torch
from mmcv.utils.logging import logger_initialized
from omegaconf import DictConfig, OmegaConf

logger_initialized["mmcv"] = logging.getLogger("mmcv")

logger = logging.getLogger(__name__)

from torch.utils.data import DataLoader

from trainers.asps import ASPSTrainer


def eval(cfg: DictConfig):
    logger.info(f"Evaluate on experiment: {cfg.experiment_dir}")

    logger.info(f"Instantiating trainer {cfg.trainer._target_}")
    trainer = hydra.utils.instantiate(cfg.trainer)

    checkpoint = trainer.checkpoint_dir

    results = []
    results.append(["Checkpoint", "Dataset", "FPS", "Mean FPS", "Dice", "IoU"])

    for ckp in os.listdir(checkpoint):
        if ckp.endswith(".pth"):
            logger.info(f"Loading checkpoint: {ckp}")
            model = torch.load(os.path.join(checkpoint, ckp))

            for dataset in cfg.datasets:
                logger.info(f"Instantiating dataloader {cfg.dataloader._target_}")
                dataloader = hydra.utils.instantiate(cfg.dataloader, dataset=dataset)

                logger.info(f"Starting evaluation {ckp}")
                fps, mean_fps, dice, iou = trainer.eval(
                    model=model,
                    dataloader=dataloader,
                    logger=logger,
                )

                dataset_name = dataset.root_dir.split("/")[-1]

                results.append([ckp, dataset_name, fps, mean_fps, dice, iou])

    with open(cfg.result_file, "w") as f:
        for result in results:
            f.write(",".join(map(str, result)) + "\n")

    logger.info("Evaluation completed")

    # logger.info(f"Instantiating model: {cfg.model._target_}")
    # model = hydra.utils.instantiate(cfg.model)

    # logger.info(f"Instantiating dataloader from dataset: {cfg.dataset._target_}")
    # dataloader = hydra.utils.instantiate(cfg.dataloader)

    # logger.info(f"Instantiating trainer ${cfg.trainer._target_}")
    # trainer = hydra.utils.instantiate(cfg.trainer)

    # logger.info(f"Starting training with {cfg.trainer.iters} iterations")
    # trainer.train(
    #     model=model,
    #     dataloader=dataloader,
    #     logger=logger,
    # )

    # logger.info("Training completed")


@hydra.main(version_base="1.3", config_path="../configs")
def main(cfg: DictConfig):
    eval(cfg)


if __name__ == "__main__":
    main()
