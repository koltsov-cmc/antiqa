import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.strategies import DDPStrategy
from lightning.pytorch.loggers import TensorBoardLogger
import gc, torch

class PLCCLoss(nn.Module):
    def __init__(self):
        super(PLCCLoss, self).__init__()
    def forward(self, input, target):
        input0 = input - torch.mean(input)
        target0 = target - torch.mean(target)
        self.loss = torch.sum(input0 * target0) / (torch.sqrt(torch.sum(input0 ** 2)) * torch.sqrt(torch.sum(target0 ** 2)))
        return self.loss
    
class DownConvGN(nn.Module):
    """Strided conv (stride=2) for downsampling + GN + ReLU."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1, bias=False),
            gn(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.proj(x)

def gn(num_channels, num_groups=8):
    # Make sure groups <= channels and divisible
    g = min(num_groups, num_channels)
    if num_channels % g != 0:
        g = 1  # fallback to LayerNorm-like behaviour
    return nn.GroupNorm(g, num_channels)

class SEBlock(nn.Module):
    def __init__(self, channels, r=16):
        super().__init__()
        hidden = max(channels // r, 8)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        w = self.gap(x).view(b, c)
        w = self.fc(w).view(b, c, 1, 1)
        return x * w

class BaseTextQualityModel(pl.LightningModule):
    """
    Parent LightningModule with:
      - combined loss: ALPHA * MSE + (1-ALPHA) * (1 - PLCC)
      - train/val/test steps
      - correlation logging
      - TensorBoard logging
      - optimizer config

    Child classes must implement:
        forward(self, imgs, ocr_scores=None) -> Tensor[B]
    """
    def __init__(self, lr: float = 1e-3, scheduler_step=10, scheduler_gamma=0.5, alpha: float = 0.5):
        super().__init__()
        self.lr = lr
        self.loss_fn = nn.MSELoss()
        self.plcc_module = PLCCLoss()

        self.alpha = float(alpha)
        self.scheduler_step = scheduler_step
        self.scheduler_gamma = scheduler_gamma

        # buffers for epoch-wise metrics
        self._val_preds = []
        self._val_targets = []

        self._val_mse_losses = []
        self._val_combined_losses = []

        self._test_preds = []
        self._test_targets = []
        self._test_combined_losses = []


    def forward(self, imgs, ocr_scores=None):
        """
        Must be implemented in child.
        Should return predictions of shape (B,) or (B, 1).
        """
        raise NotImplementedError

    # ---------------- TRAIN ----------------
    def training_step(self, batch, batch_idx):
        imgs, ocrs, ratings = batch
        preds = self(imgs, ocrs)
        preds = preds.view(-1)
        ratings = ratings.view_as(preds)

        mse = self.loss_fn(preds, ratings)


        is_mse_only = (preds.size(0) == 1)
        if is_mse_only:
            loss = mse
            plcc_val = torch.tensor(0.0, device=loss.device)
            combined_loss = loss  # for logging consistency
        else:
            plcc_val = self.plcc_module(preds, ratings)
            # numeric safety
            plcc_val = torch.clamp(plcc_val, -1.0, 1.0)
            plcc_val = torch.nan_to_num(plcc_val, nan=0.0, posinf=1.0, neginf=-1.0)
            combined_loss = self.alpha * mse + (1.0 - self.alpha) * (1.0 - plcc_val)
            loss = combined_loss


        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_mse", float(mse.detach()), on_step=True, on_epoch=True, prog_bar=False)
        self.log("train_plcc", float(plcc_val.detach()) if isinstance(plcc_val, torch.Tensor) else float(plcc_val), on_step=True, on_epoch=True, prog_bar=False)

        self.log("train_loss_is_mse_only", float(is_mse_only), on_step=True, on_epoch=True, prog_bar=False)

        return loss

    # ---------------- VALIDATION ----------------
    def on_validation_epoch_start(self) -> None:
        self._val_preds = []
        self._val_targets = []
        self._val_losses = []

    def validation_step(self, batch, batch_idx):
        imgs, ocrs, ratings = batch
        preds = self(imgs, ocrs)
        preds = preds.view(-1)
        ratings = ratings.view_as(preds)

        mse = self.loss_fn(preds, ratings)
        is_mse_only = (preds.size(0) == 1)

        if is_mse_only:
            loss = mse
            plcc_val = torch.tensor(0.0, device=loss.device)
            combined_loss = loss
        else:
            plcc_val = self.plcc_module(preds, ratings)
            plcc_val = torch.clamp(plcc_val, -1.0, 1.0)
            plcc_val = torch.nan_to_num(plcc_val, nan=0.0, posinf=1.0, neginf=-1.0)
            combined_loss = self.alpha * mse + (1.0 - self.alpha) * (1.0 - plcc_val)
            loss = combined_loss

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)


        self.log("val_mse", float(mse.detach()), on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log("val_plcc_batch", float(plcc_val.detach()) if isinstance(plcc_val, torch.Tensor) else float(plcc_val), on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log("val_combined_loss", float(combined_loss.detach()), on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log("val_loss_is_mse_only", float(is_mse_only), on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)

        self._val_preds.append(preds.detach())
        self._val_targets.append(ratings.detach())
        self._val_losses.append(loss.detach())

    def on_validation_epoch_end(self) -> None:
        if not self._val_preds:
            return

        preds = torch.cat(self._val_preds, dim=0)
        targets = torch.cat(self._val_targets, dim=0)

        if getattr(self.trainer, "world_size", 1) > 1:
            preds = self.all_gather(preds).reshape(-1)
            targets = self.all_gather(targets).reshape(-1)

        preds_cpu = preds.detach().cpu().view(-1)
        targets_cpu = targets.detach().cpu().view(-1)

        corr_res = correlations(targets_cpu, preds_cpu)  

        for k, v in corr_res.items():
            self.log(f"val_{k}", float(v), on_epoch=True, sync_dist=True, prog_bar=(k == "spearman"))

        if getattr(self.trainer, "is_global_zero", True):
            try:
                tb = self.logger.experiment
                avg_val_loss = float(torch.stack(self._val_losses).mean().item())
                tb.add_scalar("val/loss_by_epoch", avg_val_loss, self.current_epoch)
                for k, v in corr_res.items():
                    tb.add_scalar(f"val/{k}_by_epoch", float(v), self.current_epoch)
            except Exception:
                pass

        self._val_preds = []
        self._val_targets = []
        self._val_losses = []

    def on_test_epoch_start(self) -> None:
        self._test_preds = []
        self._test_targets = []

    def test_step(self, batch, batch_idx):
        imgs, ocrs, ratings = batch
        preds = self(imgs, ocrs)
        preds = preds.view(-1)
        ratings = ratings.view_as(preds)

        mse = self.loss_fn(preds, ratings)
        is_mse_only = (preds.size(0) == 1)

        if is_mse_only:
            loss = mse
            plcc_val = torch.tensor(0.0, device=loss.device)
            combined_loss = loss
        else:
            plcc_val = self.plcc_module(preds, ratings)
            plcc_val = torch.clamp(plcc_val, -1.0, 1.0)
            plcc_val = torch.nan_to_num(plcc_val, nan=0.0, posinf=1.0, neginf=-1.0)
            combined_loss = self.alpha * mse + (1.0 - self.alpha) * (1.0 - plcc_val)
            loss = combined_loss

        self.log("test_loss", loss, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("test_mse", float(mse.detach()), on_epoch=True, prog_bar=False, sync_dist=True)
        self.log("test_plcc_batch", float(plcc_val.detach()) if isinstance(plcc_val, torch.Tensor) else float(plcc_val), on_epoch=True, prog_bar=False, sync_dist=True)
        self.log("test_combined_loss", float(combined_loss.detach()), on_epoch=True, prog_bar=False, sync_dist=True)
        self.log("test_loss_is_mse_only", float(is_mse_only), on_epoch=True, prog_bar=False, sync_dist=True)

        self._test_preds.append(preds.detach())
        self._test_targets.append(ratings.detach())

        return {"test_loss": loss}

    def on_test_epoch_end(self) -> None:
        if not self._test_preds:
            return

        preds = torch.cat(self._test_preds, dim=0)
        targets = torch.cat(self._test_targets, dim=0)

        if getattr(self.trainer, "world_size", 1) > 1:
            preds = self.all_gather(preds).reshape(-1)
            targets = self.all_gather(targets).reshape(-1)

        preds_cpu = preds.detach().cpu().view(-1)
        targets_cpu = targets.detach().cpu().view(-1)

        corr_res = correlations(targets_cpu, preds_cpu)
        for k, v in corr_res.items():
            self.log(f"test_{k}", float(v), on_epoch=True, sync_dist=True)

        if getattr(self.trainer, "is_global_zero", True):
            try:
                tb = self.logger.experiment
                for k, v in corr_res.items():
                    tb.add_scalar(f"test/{k}_by_epoch", float(v), self.current_epoch)
            except Exception:
                pass

        self._test_preds = []
        self._test_targets = []

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            # decay the LR every five epochs
            step_size=self.scheduler_step,
            gamma=self.scheduler_gamma,
        )

        lr_scheduler_config = {
            "scheduler": lr_scheduler,
            "interval": "epoch",
            "frequency": 1,
        }

        return [optimizer], [lr_scheduler_config]

class ResConvBlockGN(nn.Module):
    """
    Residual block: Conv-GN-ReLU-Conv-GN (+ optional Dropout2d).
    """
    def __init__(self, channels, p_drop: float = 0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.gn1   = gn(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.gn2   = gn(channels)
        self.relu  = nn.ReLU(inplace=True)
        self.drop2d = nn.Dropout2d(p_drop) if p_drop > 0 else nn.Identity()

    def forward(self, x):
        residual = x
        out = self.relu(self.gn1(self.conv1(x)))
        out = self.gn2(self.conv2(out))
        out = self.drop2d(out)
        out = out + residual
        out = self.relu(out)
        return out


class ANTIQA(BaseTextQualityModel):
    def __init__(
        self,
        in_ch: int = 1,
        use_ocr: bool = False,
        lr: float = 1e-3,
        dropout: float = 0.2,
        grid_size: int = 2,   # <--- NEW: size of pooling grid (1 = global)
        scheduler_step=5, scheduler_gamma=0.1,
    ):
        super().__init__(lr=lr)
        self.save_hyperparameters()
        self.grid_size = grid_size

        # --- Stem: in_ch -> 64 ---
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, 64, kernel_size=3, padding=1, bias=False),
            gn(64),
            nn.ReLU(inplace=True),
        )

        # --- Stages: 64 -> 128 -> 256 ---
        # Stage 1: 64
        self.stage1 = nn.Sequential(
            ResConvBlockGN(64, p_drop=dropout * 0.5),
            ResConvBlockGN(64, p_drop=dropout * 0.5),
        )
        self.down1 = DownConvGN(64, 128)

        # Stage 2: 128
        self.stage2 = nn.Sequential(
            ResConvBlockGN(128, p_drop=dropout * 0.5),
            ResConvBlockGN(128, p_drop=dropout * 0.5),
        )
        self.down2 = DownConvGN(128, 256)

        # Stage 3: 256
        self.stage3 = nn.Sequential(
            ResConvBlockGN(256, p_drop=dropout),
            ResConvBlockGN(256, p_drop=dropout),
        )
        
        self.se0 = SEBlock(64)
        self.se1 = SEBlock(128)
        self.se2 = SEBlock(256)
        
        G = self.grid_size
        self.scale_proj0 = nn.Linear(2 * 64 * G * G, 64)
        self.scale_proj1 = nn.Linear(2 * 128 * G * G, 64)
        self.scale_proj2 = nn.Linear(2 * 256 * G * G, 64)

        self.gap = nn.AdaptiveAvgPool2d(grid_size)
        self.gmp = nn.AdaptiveMaxPool2d(grid_size)

        feat_dims = [64, 128, 256]
        self.feat_dims = feat_dims

        cells = grid_size * grid_size          # number of cells per channel
        # avg + max => factor 2
        base_feat_dim = sum(c * cells * 2 for c in feat_dims)

        feat_dim = base_feat_dim

        # --- OCR branch ---
        self.use_ocr = use_ocr
        if use_ocr:
            self.ocr_mlp = nn.Sequential(
                nn.Linear(1, 16),
                nn.ReLU(inplace=True),
                nn.Linear(16, 16),
                nn.ReLU(inplace=True),
            )
            feat_dim += 16
        else:
            self.ocr_mlp = None

        # --- Head ---
        self.head = nn.Sequential(
            nn.Linear(64*3, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.2),
            nn.Linear(32, 1),
        )

    def forward(self, imgs, ocr_scores=None):
        """
        imgs: B x C x H x W
        ocr_scores: B or B x 1 (optional)
        Returns: B (flattened regression output)
        """
        x0 = self.stem(imgs)         # B x 64 x H x W
        x0 = self.stage1(x0)         # B x 64 x H x W
        x0 = self.se0(x0)
        x1 = self.down1(x0)          # B x 128 x H/2 x W/2
        x1 = self.stage2(x1)         # B x 128 x H/2 x W/2
        x1 = self.se1(x1)
        x2 = self.down2(x1)          # B x 256 x H/4 x W/4
        x2 = self.stage3(x2)         # B x 256 x H/4 x W/4
        x2 = self.se2(x2)

        # pooling on GxG grid with avg + max
        def pool_avg_max_grid(feat):
            # feat: B x C x H x W
            avg = self.gap(feat)          # B x C x G x G
            mx  = self.gmp(feat)          # B x C x G x G
            B = feat.size(0)
            avg = avg.view(B, -1)         # B x (C * G * G)
            mx  = mx.view(B, -1)          # B x (C * G * G)
            return torch.cat([avg, mx], dim=1)  # B x (2 * C * G * G)

        f0 = self.scale_proj0(pool_avg_max_grid(x0)) # B x 64
        f1 = self.scale_proj1(pool_avg_max_grid(x1)) # B x 64
        f2 = self.scale_proj2(pool_avg_max_grid(x2)) # B x 64

        x = torch.cat([f0, f1, f2], dim=1)  # B x 64*3

        # OCR branch
        if self.use_ocr:
            if ocr_scores is None:
                o = torch.zeros(x.size(0), 1, device=x.device, dtype=x.dtype)
            else:
                o = ocr_scores.view(-1, 1).to(device=x.device, dtype=x.dtype)
            o = self.ocr_mlp(o)       # B x 16
            x = torch.cat([x, o], dim=1)

        out = self.head(x)            # B x 1
        return out.view(-1)
