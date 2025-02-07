import rootutils

rootutils.setup_root(__file__, ".project-root", pythonpath=True)

import logging

import hydra
from mmcv.utils.logging import logger_initialized
from omegaconf import DictConfig, OmegaConf

logger_initialized["mmcv"] = logging.getLogger("mmcv")

logger = logging.getLogger(__name__)

from torch.utils.data import DataLoader

from trainer.asps import ASPSTrainer


def train(cfg: DictConfig):
    logger.info(f"Instantiating model: {cfg.model._target_}")
    model = hydra.utils.instantiate(cfg.model)

    logger.info(f"Instantiating dataset: {cfg.dataset._target_}")
    dataset = hydra.utils.instantiate(cfg.dataset)

    logger.info(f"Instantiating optimizer: {cfg.trainer.optimizer._target_}")
    optimizer = hydra.utils.instantiate(cfg.trainer.optimizer, model.parameters())

    logger.info("Converting the dataset to dataloader")
    train_loader = DataLoader(
        dataset=dataset,
        batch_size=cfg.trainer.batch_size,
        shuffle=True,
        num_workers=cfg.trainer.num_workers,
        pin_memory=True,
    )

    logger.info("Instantiating trainer ASPSTrainer")
    trainer = ASPSTrainer(
        model=model,
        optimizer=optimizer,
        device=cfg.trainer.device,
    )

    logger.info(f"Starting training with {cfg.trainer.iters} iterations")
    trainer.train(
        dataloader=train_loader,
        iters=cfg.trainer.iters,
        save_iter=cfg.trainer.save_iter,
        checkpoint_dir=cfg.trainer.checkpoint_dir,
        budget=cfg.trainer.budget,
        clipping=cfg.trainer.clipping,
        logger=logger,
        tensorboard_path=cfg.trainer.tensorboard_path,
    )

    logger.info("Training completed")


@hydra.main(version_base="1.3", config_path="../configs")
def main(cfg: DictConfig):
    train(cfg)


if __name__ == "__main__":
    main()
