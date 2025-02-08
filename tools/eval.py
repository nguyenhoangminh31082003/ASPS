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
from prettytable import PrettyTable

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

    for ckp in sorted(os.listdir(checkpoint)):
        if ckp.endswith(".pth"):
            logger.info(f"Loading checkpoint: {ckp}")
            model = torch.load(os.path.join(checkpoint, ckp))
            model = model.to(trainer.device)

            for dataset in cfg.datasets:
                logger.info(f"Evaluating with {dataset.root_dir}")
                dataloader = hydra.utils.instantiate(cfg.dataloader, dataset=dataset)

                fps, mean_fps, dice, iou = trainer.eval(
                    model=model,
                    dataloader=dataloader,
                )

                dataset_name = dataset.root_dir.split("/")[-1]

                results.append([ckp, dataset_name, fps, mean_fps, dice, iou])
                logger.info(
                    f"Checkpoint: {ckp}, Dataset: {dataset_name}, FPS: {fps}, Mean FPS: {mean_fps}, Dice: {dice}, IoU: {iou}"
                )

    table = PrettyTable()
    table.field_names = results[0]
    for row in results[1:]:
        table.add_row(row)
    logger.info(f"Results:\n{table}")

    with open(cfg.result_file, "w") as f:
        for result in results:
            f.write(",".join(map(str, result)) + "\n")

    logger.info("Evaluation completed")


@hydra.main(version_base="1.3", config_path="../configs")
def main(cfg: DictConfig):
    eval(cfg)


if __name__ == "__main__":
    main()
