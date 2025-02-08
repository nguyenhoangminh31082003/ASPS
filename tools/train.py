import rootutils

rootutils.setup_root(__file__, ".project-root", pythonpath=True)

import warnings

warnings.filterwarnings("ignore")

import logging

import hydra
from mmcv.utils.logging import logger_initialized
from omegaconf import DictConfig, OmegaConf

logger_initialized["mmcv"] = logging.getLogger("mmcv")

logger = logging.getLogger(__name__)

from torch.utils.data import DataLoader

from trainers.asps import ASPSTrainer


def train(cfg: DictConfig):
    logger.info(f"Instantiating model: {cfg.model._target_}")
    model = hydra.utils.instantiate(cfg.model)

    # logger.info(f"Instantiating dataset: {cfg.dataset._target_}")
    # dataset = hydra.utils.instantiate(cfg.dataset)

    logger.info(f"Instantiating dataloader from dataset: {cfg.dataset._target_}")
    dataloader = hydra.utils.instantiate(cfg.dataloader)

    # logger.info("Converting the dataset to dataloader")
    # train_loader = DataLoader(
    #     dataset=dataset,
    #     batch_size=cfg.trainer.batch_size,
    #     shuffle=True,
    #     num_workers=cfg.trainer.num_workers,
    #     pin_memory=True,
    # )

    logger.info(f"Instantiating trainer ${cfg.trainer._target_}")
    trainer = hydra.utils.instantiate(cfg.trainer)

    logger.info(f"Starting training with {cfg.trainer.iters} iterations")
    trainer.train(
        model=model,
        dataloader=dataloader,
        logger=logger,
    )

    logger.info("Training completed")


@hydra.main(version_base="1.3", config_path="../configs")
def main(cfg: DictConfig):
    train(cfg)


if __name__ == "__main__":
    main()
