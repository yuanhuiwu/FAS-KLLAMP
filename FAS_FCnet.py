# FAS200_FCNet_train.py
# -*- coding: utf-8 -*-
"""
Train FCNet for FAS channel estimation
Supports:
  1) Multiple scenarios (Rayleigh, Rician K=5/10/15 dB)
  2) Enhanced network architecture (residual connections + Batch Normalization)
  3) Training SNR range 0~30 dB
"""
import os
import math
import json
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# --- Import utilities from single-user LAMP file ---
from FAS_KLLAMP import (
    GenCfg,
    build_npz_from_fas,
    LAMPDataset,
    nmse_db_h_from_coeff,
    get_scenario_configs,
)


# =========================================================
# Checkpoint Utilities
# =========================================================
def save_checkpoint(path: str, model: nn.Module,
                    optimizer: torch.optim.Optimizer,
                    epoch: int, best_val_loss: float):
    """Save training checkpoint."""
    ckpt = {
        "epoch": epoch,
        "best_val_loss": best_val_loss,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(ckpt, path)


def load_checkpoint(path: str, model: nn.Module,
                    optimizer: Optional[torch.optim.Optimizer],
                    map_location: str):
    """Load training checkpoint."""
    ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt["model"])
    if optimizer is not None and "optimizer" in ckpt:
        try:
            optimizer.load_state_dict(ckpt["optimizer"])
        except Exception as e:
            print(f"[Warning] Optimizer state mismatch: {e}")
    epoch = int(ckpt.get("epoch", ckpt.get("best_val_loss", 0)))
    best_val_loss = float(ckpt.get("best_val_loss", ckpt.get("best_val", 1e9)))
    return epoch, best_val_loss, ckpt


# =========================================================
# Scenario File Path Management
# =========================================================
@dataclass
class ScenarioFiles:
    """File paths for a single scenario."""
    name: str
    k_factor_db: float
    npz_path: str
    lamp_ckpt: str
    fcnet_ckpt: str

    @property
    def display_name(self) -> str:
        if self.k_factor_db < -50:
            return "Rayleigh"
        else:
            return f"Rician K={self.k_factor_db:.0f}dB"


def get_scenario_file_paths(D: int = 12, M: int = 8, Tpilot: int = 2,
                            base_dir: str = ".") -> Dict[str, ScenarioFiles]:
    """Get file paths for all scenarios."""
    scenarios = {}

    scenario_defs = [
        ("rayleigh", -np.inf),
        ("rician_5dB", 5.0),
        ("rician_10dB", 10.0),
        ("rician_15dB", 15.0),
    ]

    for name, k_db in scenario_defs:
        if k_db < -50:
            tag = f"D{D}_M{M}_T{Tpilot}_rayleigh_snr0to30"
        else:
            tag = f"D{D}_M{M}_T{Tpilot}_K{k_db:.0f}dB_snr0to30"

        scenarios[name] = ScenarioFiles(
            name=name,
            k_factor_db=k_db,
            npz_path=os.path.join(base_dir, f"fas_lamp_dataset_{tag}.npz"),
            lamp_ckpt=os.path.join(base_dir, f"lamp_fas_best_{tag}.pth"),
            fcnet_ckpt=os.path.join(base_dir, f"fcnet_fas_best_{tag}.pth"),
        )

    return scenarios


# =========================================================
# Enhanced FCNet (Residual Blocks + Batch Normalization + Dropout)
# =========================================================
class ResidualBlock(nn.Module):
    """Fully connected residual block."""
    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.bn1 = nn.BatchNorm1d(dim)
        self.fc2 = nn.Linear(dim, dim)
        self.bn2 = nn.BatchNorm1d(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.fc1(x)))
        out = self.dropout(out)
        out = self.bn2(self.fc2(out))
        out = out + residual
        out = F.relu(out)
        return out


class FCNetEnhanced(nn.Module):
    """
    Enhanced fully connected neural network.
    Features:
      - Residual connections
      - Batch normalization
      - Dropout regularization
    """
    def __init__(self, in_dim: int, out_dim: int,
                 hidden_dim: int = 512,
                 num_res_blocks: int = 3,
                 dropout: float = 0.1):
        super().__init__()

        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        res_blocks = []
        for _ in range(num_res_blocks):
            res_blocks.append(ResidualBlock(hidden_dim, dropout=dropout))
        self.res_blocks = nn.Sequential(*res_blocks)

        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(256, out_dim)
        )

    def forward(self, y: torch.Tensor):
        B, M, _ = y.shape
        x = y.view(B, M)
        x = self.input_proj(x)
        x = self.res_blocks(x)
        x = self.output_proj(x)
        return x.unsqueeze(2)


class FCNet(nn.Module):
    """Basic fully connected neural network."""
    def __init__(self, in_dim: int, out_dim: int, hidden_dims=(256, 256)):
        super().__init__()
        layers = []
        last = in_dim
        for h in hidden_dims:
            layers.append(nn.Linear(last, h))
            layers.append(nn.ReLU())
            last = h
        layers.append(nn.Linear(last, out_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, y: torch.Tensor):
        B, M, _ = y.shape
        inp = y.view(B, M)
        out = self.mlp(inp)
        return out.unsqueeze(2)


# =========================================================
# Single Scenario Training Function
# =========================================================
def train_fcnet_single(
        npz_path: str,
        device: str = "cuda",
        total_epochs: int = 200,
        batch_size: int = 256,
        lr: float = 2e-3,
        weight_decay: float = 1e-4,
        resume_path: str = "fcnet_last_ckpt.pth",
        best_path: str = "fcnet_fas_best.pth",
        snr_db_min_train: float = 0.0,
        snr_db_max_train: float = 30.0,
        snr_db_val: float = 10.0,
        scenario_tag: str = "",
        use_enhanced: bool = True):
    """
    Train FCNet for a single scenario.
    
    Args:
        npz_path: Path to dataset
        device: Computing device ('cuda' or 'cpu')
        total_epochs: Number of training epochs
        batch_size: Training batch size
        lr: Learning rate
        weight_decay: Weight decay for optimizer
        resume_path: Path to save/load last checkpoint
        best_path: Path to save best model
        snr_db_min_train: Minimum training SNR (dB)
        snr_db_max_train: Maximum training SNR (dB)
        snr_db_val: Validation SNR (dB)
        scenario_tag: Label for this scenario
        use_enhanced: Whether to use enhanced architecture
    """
    print(f"\n{'=' * 60}")
    print(f"Training FCNet — {scenario_tag}")
    print(f"  Dataset : {npz_path}")
    print(f"  Network : {'Enhanced' if use_enhanced else 'Basic'}")
    print(f"  SNR train: [{snr_db_min_train}, {snr_db_max_train}] dB")
    print(f"  SNR val  : {snr_db_val} dB")
    print(f"  Epochs   : {total_epochs}")
    print(f"  Best path: {best_path}")
    print(f"{'=' * 60}\n")

    # Build datasets
    train_ds = LAMPDataset(
        npz_path, split="train", device=device,
        online_noise=True,
        snr_db_min=snr_db_min_train,
        snr_db_max=snr_db_max_train
    )
    val_ds = LAMPDataset(
        npz_path, split="val", device=device,
        online_noise=True,
        snr_db_min=snr_db_val,
        snr_db_max=snr_db_val
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=512, shuffle=False)

    # Dimension information
    D = train_ds.X_real.shape[1] // 2
    
    if train_ds.Y_real is not None:
        m2 = train_ds.Y_real.shape[1]
    else:
        m2 = 2 * train_ds.m
    
    in_dim = m2
    out_dim = 2 * D

    print(f"  in_dim (2m) = {in_dim}")
    print(f"  out_dim (2D) = {out_dim}")

    U = train_ds.U.to(device=device, dtype=torch.complex64)

    # Create model
    if use_enhanced:
        model = FCNetEnhanced(
            in_dim=in_dim,
            out_dim=out_dim,
            hidden_dim=512,
            num_res_blocks=3,
            dropout=0.1
        ).to(device)
    else:
        model = FCNet(
            in_dim=in_dim,
            out_dim=out_dim,
            hidden_dims=(256, 256)
        ).to(device)

    print(f"\nModel architecture:")
    print(model)
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Learning rate scheduler: reduce LR on plateau
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode='min', factor=0.5, patience=10
    )
    crit = nn.MSELoss()

    start_epoch = 1
    best_val = 1e9
    prev_lr = lr  # Track LR changes manually

    # Resume from checkpoint if exists
    if resume_path is not None and os.path.exists(resume_path):
        try:
            ep, best_val, ckpt = load_checkpoint(
                resume_path, model, opt, map_location=device
            )
            start_epoch = ep + 1
            print(f"[FCNet-Resume] Loaded {resume_path}, "
                  f"start_epoch={start_epoch}, best_val={best_val:.4e}")
        except Exception as e:
            print(f"[FCNet-Resume] Failed to load checkpoint: {e}")
            print("  Starting from scratch...")

    # Training loop
    for ep in range(start_epoch, total_epochs + 1):
        # ---------- Training ----------
        model.train()
        tr_loss = 0.0
        for Y, Xgt, nv, Ht in train_loader:
            Y = Y.to(device)
            Xgt = Xgt.to(device)

            x_hat = model(Y)
            loss = crit(x_hat, Xgt)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()

            tr_loss += float(loss.item())
        tr_loss /= max(1, len(train_loader))

        # ---------- Validation ----------
        model.eval()
        va_loss = 0.0
        va_nmse_list = []
        with torch.no_grad():
            for Y, Xgt, nv, Ht in val_loader:
                Y = Y.to(device)
                Xgt = Xgt.to(device)
                Ht = Ht.to(device)

                x_hat = model(Y)
                va_loss += float(crit(x_hat, Xgt).item())

                nmse_h = nmse_db_h_from_coeff(x_hat, Ht, U, D)
                va_nmse_list.append(nmse_h)

        va_loss /= max(1, len(val_loader))
        va_nmse = float(np.mean(va_nmse_list))

        # Learning rate scheduling
        scheduler.step(va_loss)
        
        # Detect and report LR changes
        current_lr = opt.param_groups[0]['lr']
        lr_changed = ""
        if current_lr < prev_lr:
            lr_changed = f" (LR reduced: {prev_lr:.2e} -> {current_lr:.2e})"
            prev_lr = current_lr

        print(f"[{scenario_tag}] Epoch {ep:03d} | "
              f"TrLoss={tr_loss:.4e} | ValLoss={va_loss:.4e} | "
              f"NMSE(h)={va_nmse:.2f} dB | "
              f"LR={current_lr:.2e}{lr_changed}")

        # Save last checkpoint
        if resume_path is not None:
            save_checkpoint(resume_path, model, opt, ep, best_val)

        # Save best model
        if va_loss < best_val:
            best_val = va_loss
            torch.save(model.state_dict(), best_path)
            if resume_path is not None:
                save_checkpoint(resume_path, model, opt, ep, best_val)
            print(f"  -> saved best to {best_path} (ValLoss={best_val:.4e})")

    print(f"\n[{scenario_tag}] Training done. Best val loss = {best_val:.4e}")
    return model


# =========================================================
# Multi-Scenario Batch Training
# =========================================================
def train_fcnet_all_scenarios(
        D: int = 12,
        M: int = 8,
        Tpilot: int = 2,
        S: int = 20000,
        total_epochs: int = 200,
        device: str = "cuda",
        use_enhanced: bool = True):
    """
    Train FCNet for all scenarios.
    
    Args:
        D: KL basis dimension
        M: Ports sampled per slot
        Tpilot: Number of pilot slots
        S: Number of samples
        total_epochs: Number of training epochs
        device: Computing device
        use_enhanced: Whether to use enhanced architecture
    
    Returns:
        Dictionary of saved file paths for each scenario
    """
    scenarios = get_scenario_configs(D=D, M=M, Tpilot=Tpilot, S=S)

    saved_files = {}

    for name, cfg in scenarios.items():
        tag = cfg.make_tag()

        npz_path = f"fas_lamp_dataset_{tag}.npz"
        resume_path = f"fcnet_last_ckpt_{tag}.pth"
        best_path = f"fcnet_fas_best_{tag}.pth"

        # 1) Check if dataset exists
        if not os.path.exists(npz_path):
            print(f"\n[{name}] Dataset not found: {npz_path}")
            print(f"  Building dataset...")
            build_npz_from_fas(npz_path, cfg)

        print(f"\n[{name}] Using dataset: {npz_path}")

        # 2) Train FCNet
        train_fcnet_single(
            npz_path=npz_path,
            device=device,
            total_epochs=total_epochs,
            batch_size=256,
            lr=2e-3,
            weight_decay=1e-4,
            resume_path=resume_path,
            best_path=best_path,
            snr_db_min_train=cfg.snr_db_min,
            snr_db_max_train=cfg.snr_db_max,
            snr_db_val=10.0,
            scenario_tag=name,
            use_enhanced=use_enhanced,
        )

        saved_files[name] = {
            "npz": npz_path,
            "resume": resume_path,
            "best": best_path,
            "config": cfg,
        }

    # Print summary
    print("\n" + "=" * 70)
    print("All FCNet scenarios completed!")
    print("=" * 70)
    for name, info in saved_files.items():
        print(f"\n[{name}]")
        print(f"  Dataset   : {info['npz']}")
        print(f"  Last ckpt : {info['resume']}")
        print(f"  Best ckpt : {info['best']}")
        k_db = info['config'].k_factor_db
        print(f"  K-factor  : {k_db} dB"
              + (" (Rayleigh)" if k_db < -50 else " (Rician)"))

    return saved_files


# =========================================================
# Compatibility: Legacy train_fcnet Interface
# =========================================================
def train_fcnet(npz_path: str,
                device: str = "cuda",
                total_epochs: int = 100,
                batch_size: int = 256,
                lr: float = 2e-3,
                weight_decay: float = 1e-4,
                resume_path: str = "fcnet_last_ckpt.pth",
                best_path: str = "fcnet_fas_best.pth"):
    """Legacy interface for backward compatibility."""
    return train_fcnet_single(
        npz_path=npz_path,
        device=device,
        total_epochs=total_epochs,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        resume_path=resume_path,
        best_path=best_path,
        snr_db_min_train=0.0,
        snr_db_max_train=30.0,
        snr_db_val=10.0,
        scenario_tag="default",
        use_enhanced=False,
    )


# =========================================================
# Main Entry Point
# =========================================================
if __name__ == "__main__":
    import argparse

    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    parser = argparse.ArgumentParser(description="FCNet training for FAS channel estimation")
    parser.add_argument("--mode", type=str, default="all",
                        choices=["all", "single"],
                        help="'all': train all scenarios; "
                             "'single': train single Rayleigh scenario")
    parser.add_argument("--epochs", type=int, default=200,
                        help="Number of training epochs")
    parser.add_argument("--D", type=int, default=12,
                        help="KL basis dimension")
    parser.add_argument("--M", type=int, default=8,
                        help="Ports sampled per slot")
    parser.add_argument("--T", type=int, default=2,
                        help="Number of pilot slots")
    parser.add_argument("--S", type=int, default=20000,
                        help="Number of samples")
    parser.add_argument("--enhanced", action='store_true', default=True,
                        help="Use enhanced FCNet architecture (default)")
    parser.add_argument("--basic", dest='enhanced', action='store_false',
                        help="Use basic FCNet architecture")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Network type: {'Enhanced' if args.enhanced else 'Basic'}")

    if args.mode == "all":
        # Train all scenarios
        saved = train_fcnet_all_scenarios(
            D=args.D,
            M=args.M,
            Tpilot=args.T,
            S=args.S,
            total_epochs=args.epochs,
            device=device,
            use_enhanced=args.enhanced,
        )

    else:
        # Train single Rayleigh scenario only
        cfg = GenCfg(
            S=args.S, N=200,
            aoa_mean_deg=20.0, aoa_spread_deg=10.0,
            D=args.D, M=args.M, Tpilot=args.T,
            snr_db_min=0.0, snr_db_max=30.0,
            seed=0,
        )
        tag = cfg.make_tag()
        npz_path = f"fas_lamp_dataset_{tag}.npz"

        if not os.path.exists(npz_path):
            print(f"\nDataset not found: {npz_path}")
            print("Building dataset...")
            build_npz_from_fas(npz_path, cfg)

        train_fcnet_single(
            npz_path=npz_path,
            device=device,
            total_epochs=args.epochs,
            batch_size=256,
            lr=2e-3,
            weight_decay=1e-4,
            resume_path=f"fcnet_last_ckpt_{tag}.pth",
            best_path=f"fcnet_fas_best_{tag}.pth",
            snr_db_min_train=0.0,
            snr_db_max_train=30.0,
            snr_db_val=10.0,
            scenario_tag="single_rayleigh",
            use_enhanced=args.enhanced,
        )