import rootutils

rootutils.setup_root(__file__, pythonpath=True)

import hydra
import torch
from omegaconf import DictConfig

from utils import seed_everything


@hydra.main(version_base="1.3", config_path="../config", config_name="train")
def main(cfg: DictConfig):
    if cfg.get("seed"):
        seed_everything(cfg.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = hydra.utils.instantiate(cfg.model)
    model = model.to(device)


if __name__ == "__main__":
    main()
