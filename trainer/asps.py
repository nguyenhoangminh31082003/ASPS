import os
import time

import torch.nn as nn
import torch.optim
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torchvision.transforms import Resize
from tqdm import tqdm

from datasets import sample_data
from losses import DiceLoss
from models.asps import ASPS

from .utils import compute_dice, compute_iou


class ASPSTrainer:
    def __init__(
        self,
        model: ASPS,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
    ):
        self.model = model
        self.optimizer = optimizer
        self.device = device

        self.mse_loss = nn.MSELoss()
        self.ce_loss = nn.CrossEntropyLoss()
        self.dice_loss = DiceLoss()

    def seg_loss(self, pred, gt):
        loss = self.ce_loss(pred, gt) + 0.5 * self.dice_loss(pred, gt) + self.mse_loss(pred, gt)
        return loss

    def train(
        self,
        dataloader,
        iters,
        checkpoint_dir,
        save_iter,
        budget=0.3,
        clipping=2,
        logger=None,
        tensorboard_path=None,
    ):
        if os.path.exists(checkpoint_dir) and os.listdir(checkpoint_dir):
            raise ValueError("Checkpoint directory already exists")
        else:
            os.makedirs(checkpoint_dir, exist_ok=True)

        lmbda = 0.1
        self.model.to(self.device)

        dataloader = sample_data(dataloader)
        writer = SummaryWriter(tensorboard_path) if tensorboard_path else None

        pbar = tqdm(range(1, iters + 1))

        for itr in pbar:
            self.model.train()

            img, gt, _ = next(dataloader)
            img, gt = img.to(self.device), gt.to(self.device)

            # Resize gt to match the mask output
            gt = torch.stack([Resize(256)(x) for x in gt])

            pred, iou_pred, uncertainty_p = self.model(img, multimask_output=False)

            # U_p + U_i
            ones = torch.ones_like(iou_pred)
            confidence = (iou_pred + (ones - uncertainty_p.mean(dim=(2, 3)))) / 2

            eps = 1e-23
            pred = torch.clamp(pred, eps, 1 - eps)
            confidence = torch.clamp(confidence, eps, 1 - eps)

            # Hint module
            b = torch.bernoulli(torch.Tensor(confidence.size()).uniform_(0, 1))
            b = b.to(self.device)

            pred_new = (1 - b[:, :, None, None]) * pred + b[:, :, None, None] * (
                confidence[:, :, None, None] * pred + (1 - confidence[:, :, None, None]) * gt
            )

            self.optimizer.zero_grad()
            confidence_loss = torch.mean(-torch.log(confidence))
            loss = self.seg_loss(pred_new, gt) + lmbda * confidence_loss

            if budget > confidence_loss.mean():
                lmbda /= 0.1
            else:
                lmbda /= 0.99

            loss.backward()

            nn.utils.clip_grad_norm_(self.model.parameters(), clipping)
            self.optimizer.step()
            pbar.set_postfix(loss=loss.item())
            pbar.update(1)

            if writer:
                writer.add_scalar("loss", loss.item(), itr)
                writer.add_scalar("confidence_loss", confidence_loss.item(), itr)

            if itr >= save_iter * 25 and (itr % save_iter == 0 or iter == iters - 1):
                torch.save(
                    self.model.state_dict(),
                    f"{checkpoint_dir}/model_{str(iter).zfill(7)}.pth",
                )

                if logger:
                    logger.info(f"Model saved at iteration {itr}")

        if writer:
            writer.close()

    def test(self, dataloader):
        self.model.eval()

        fps = []
        num_frames = 0
        dice, iou = [], []

        t0 = time.time()
        for img, gt, _ in dataloader:
            img, gt = img.to(self.device), gt.to(self.device)

            with torch.no_grad():
                t1 = time.time()
                pred, _, _ = self.model(img, multimask_output=False)
                fps.append(img.size(0) / (time.time() - t1))

            pred = pred > 0.5
            t2 = time.time()
            num_frames += img.size(0)
            dice.append(compute_dice(pred, gt).item())
            iou.append(compute_iou(pred, gt).item())

        fps_save = num_frames / (t2 - t0)

        return (
            sum(fps) / len(fps),
            fps_save,
            "%.4f" % (sum(dice) / len(dice)),
            "%.4f" % (sum(iou) / len(iou)),
        )
