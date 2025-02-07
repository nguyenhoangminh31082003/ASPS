import rootutils
import sys

rootutils.setup_root(__file__, pythonpath=True)

import hydra
import torch
from omegaconf import DictConfig

from utils import seed_everything

# Print sys.path to verify that your project root is included
print("sys.path:", sys.path)

# Try importing ASPS from the model package and print the result
try:
    from model import ASPS
    print("Successfully imported model.ASPS:", ASPS)
except Exception as e:
    print("Error importing model.ASPS:", e)

@hydra.main(version_base="1.3", config_path="../config", config_name="train")
def main(cfg: DictConfig):
    if cfg.get("seed"):
        seed_everything(cfg.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Device: {device}")
    print(f"Config: {cfg}")
    print(f"Model: {cfg.model}")

    model = hydra.utils.instantiate(cfg.model)
    model = model.to(device)


if __name__ == "__main__":
    main()
