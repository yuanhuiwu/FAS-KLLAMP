# -*- coding: utf-8 -*-
"""
FAS200_LAMP.py — Complete LAMP training and evaluation code for 200-port FAS channel (single user)
New features:
  1) Training SNR range 0~30 dB
  2) Rician fading scenario training and saving
  3) Separate dataset and weights for multiple scenarios
"""
import os, math, json
import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, Optional, Dict, Any, Union, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from scipy.linalg import eigh

from FAS_Channel_Model import FASChannelWithAoA


# =========================================================
# 1) AoA -> Theoretical Covariance R -> KL Basis U_D
# =========================================================
def build_R_directional(x: np.ndarray, theta0: float, sigma: float,
                        k: float = 2 * np.pi) -> np.ndarray:
    """Build correlation matrix R under directional Gaussian AoA distribution."""
    X = x[:, None]
    d = X - X.T
    phase = np.exp(1j * k * d * np.sin(theta0))
    decay = np.exp(-0.5 * (k * np.abs(d) * np.cos(theta0) * sigma) ** 2)
    R = phase * decay
    R = (R + R.conj().T) / 2
    np.fill_diagonal(R, 1.0 + 0j)
    return R


def kl_basis_from_R(R: np.ndarray, D: int) -> np.ndarray:
    """Compute KL basis U_D from correlation matrix R."""
    w, V = eigh(R)
    idx = np.argsort(w)[::-1][:D]
    U = V[:, idx]
    return U.astype(np.complex128)


# =========================================================
# 2) Port Selection Schedule (Pilot Sampling)
# =========================================================
def make_block_hopping_indices(N: int, M: int, T: int,
                               seed: int = 0) -> np.ndarray:
    """
    Generate block-hopping pilot indices.
    Total measurements m = M*T. Each slot picks one port per block with hopping.
    """
    assert 1 <= M <= N
    rng = np.random.default_rng(seed)

    edges = np.linspace(0, N, M + 1).astype(int)
    blocks = [np.arange(edges[i], edges[i + 1]) for i in range(M)]

    idx_all = []
    for _ in range(T):
        perm = rng.permutation(M)
        for b in perm:
            idx_all.append(int(rng.choice(blocks[b])))
    return np.array(idx_all, dtype=np.int64)


# =========================================================
# 3) Complex -> Real-Stacked Linear System
# =========================================================
def complex_to_real_system(A_c: np.ndarray,
                           y_c: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Convert complex linear system to real-valued stacked form."""
    Ar, Ai = A_c.real, A_c.imag
    top = np.concatenate([Ar, -Ai], axis=1)
    bot = np.concatenate([Ai, Ar], axis=1)
    A_real = np.concatenate([top, bot], axis=0)
    y_real = np.concatenate([y_c.real, y_c.imag], axis=0)[:, None]
    return A_real.astype(np.float32), y_real.astype(np.float32)


# =========================================================
# 4) Dataset Configuration (Multi-Scenario Support)
# =========================================================
@dataclass
class GenCfg:
    """Dataset generation configuration."""
    S: int = 20000                      # Number of samples
    N: int = 200                        # Number of FAS ports
    W_lambda: float = 4.0               # FAS length in wavelengths
    fc_ghz: float = 3.5                 # Carrier frequency (GHz)
    aoa_mean_deg: float = 20.0          # Mean AoA (degrees)
    aoa_spread_deg: float = 10.0        # Angular spread std (degrees)
    use_isotropic: bool = False         # Use isotropic model
    k_factor_db: float = -np.inf        # Rician K-factor (dB), -inf for Rayleigh

    D: int = 12                         # KL basis dimension
    M: int = 8                          # Ports sampled per slot
    Tpilot: int = 2                     # Number of pilot slots

    # Training SNR range
    snr_db_min: float = 0.0
    snr_db_max: float = 30.0

    save_y_clean: bool = True           # Save clean observations
    seed: int = 0

    # Scenario label (for identification)
    scenario_tag: str = "rayleigh"

    def make_tag(self) -> str:
        """Generate filename tag."""
        tag = f"D{self.D}_M{self.M}_T{self.Tpilot}"
        if self.k_factor_db > -50:
            tag += f"_K{self.k_factor_db:.0f}dB"
        else:
            tag += "_rayleigh"
        tag += f"_snr{self.snr_db_min:.0f}to{self.snr_db_max:.0f}"
        return tag


# =========================================================
# 5) Pre-Defined Training Scenarios
# =========================================================
def get_scenario_configs(D: int = 12, M: int = 8, Tpilot: int = 2,
                         S: int = 20000) -> Dict[str, GenCfg]:
    """
    Return configuration dictionary for multiple scenarios:
      - rayleigh:     Rayleigh fading (K = -inf dB)
      - rician_5dB:   Rician fading K = 5 dB
      - rician_10dB:  Rician fading K = 10 dB
      - rician_15dB:  Rician fading K = 15 dB
    All scenarios use SNR training range 0~30 dB.
    """
    base_kwargs = dict(
        S=S, N=200, W_lambda=4.0, fc_ghz=3.5,
        aoa_mean_deg=20.0, aoa_spread_deg=10.0,
        use_isotropic=False,
        D=D, M=M, Tpilot=Tpilot,
        snr_db_min=0.0, snr_db_max=30.0,
        save_y_clean=True,
        seed=0,
    )

    scenarios = {
        "rayleigh": GenCfg(
            **base_kwargs,
            k_factor_db=-np.inf,
            scenario_tag="rayleigh",
        ),
        "rician_5dB": GenCfg(
            **base_kwargs,
            k_factor_db=5.0,
            scenario_tag="rician_5dB",
        ),
        "rician_10dB": GenCfg(
            **base_kwargs,
            k_factor_db=10.0,
            scenario_tag="rician_10dB",
        ),
        "rician_15dB": GenCfg(
            **base_kwargs,
            k_factor_db=15.0,
            scenario_tag="rician_15dB",
        ),
    }
    return scenarios


# =========================================================
# 6) Dataset Building (Rician Fading Support)
# =========================================================
def build_npz_from_fas(save_path: str, cfg: GenCfg):
    """Generate FAS channel dataset and save to npz file."""
    rng = np.random.default_rng(cfg.seed)

    print(f"\n{'=' * 60}")
    print(f"Building dataset: {save_path}")
    print(f"  Scenario     : {cfg.scenario_tag}")
    print(f"  K-factor     : {cfg.k_factor_db} dB")
    print(f"  SNR range    : [{cfg.snr_db_min}, {cfg.snr_db_max}] dB")
    print(f"  Samples      : {cfg.S}")
    print(f"  D={cfg.D}, M={cfg.M}, T={cfg.Tpilot}")
    print(f"{'=' * 60}")

    # Generate FAS channel
    gen = FASChannelWithAoA(
        num_ports=cfg.N,
        fas_length_lambda=cfg.W_lambda,
        fc_ghz=cfg.fc_ghz,
        min_dist=50, max_dist=500,
        use_isotropic=cfg.use_isotropic,
        aoa_mean_deg=cfg.aoa_mean_deg,
        aoa_spread_deg=cfg.aoa_spread_deg,
        k_factor_db=cfg.k_factor_db,
    )

    # Generate channel samples
    _, H_small, _, _ = gen.generate(cfg.S, return_small=True)
    H = H_small.astype(np.complex128)

    # Construct KL basis
    theta0 = np.deg2rad(cfg.aoa_mean_deg)
    sigma = np.deg2rad(cfg.aoa_spread_deg)
    R = build_R_directional(gen.x, theta0, sigma, k=gen.k)
    U = kl_basis_from_R(R, cfg.D)

    # Port sampling indices
    idx_all = make_block_hopping_indices(cfg.N, cfg.M, cfg.Tpilot, seed=cfg.seed)
    m = idx_all.shape[0]

    # Measurement matrix (real domain)
    A_c = U[idx_all, :]
    A_real, _ = complex_to_real_system(A_c, np.zeros((m,), np.complex128))

    Y_real = np.zeros((cfg.S, 2 * m, 1), dtype=np.float32)
    X_real = np.zeros((cfg.S, 2 * cfg.D, 1), dtype=np.float32)
    noise_var_real = np.zeros((cfg.S,), dtype=np.float32)
    snr_db_arr = np.zeros((cfg.S,), dtype=np.float32)

    Y_clean_c = np.zeros((cfg.S, m), dtype=np.complex64) if cfg.save_y_clean else None

    for s in range(cfg.S):
        h = H[s]
        y_clean = h[idx_all]
        sig_pow = np.mean(np.abs(y_clean) ** 2) + 1e-12

        # Random SNR per sample
        snr_db_s = float(rng.uniform(cfg.snr_db_min, cfg.snr_db_max))
        snr_lin = 10 ** (snr_db_s / 10.0)
        sigma2_complex = sig_pow / snr_lin
        sigma2_real = sigma2_complex / 2.0

        n = (rng.standard_normal(m) + 1j * rng.standard_normal(m)) * math.sqrt(sigma2_complex / 2.0)
        y = y_clean + n

        # Ground truth coefficients
        c = U.conj().T @ h
        y_real = np.concatenate([y.real, y.imag], axis=0)[:, None].astype(np.float32)

        Y_real[s] = y_real
        X_real[s] = np.concatenate([c.real, c.imag], axis=0)[:, None].astype(np.float32)
        noise_var_real[s] = np.float32(sigma2_real)
        snr_db_arr[s] = np.float32(snr_db_s)
        if Y_clean_c is not None:
            Y_clean_c[s] = y_clean.astype(np.complex64)

    out = dict(
        A_real=A_real,
        Y_real=Y_real,
        X_real=X_real,
        H_true=H.astype(np.complex64),
        U=U.astype(np.complex64),
        idx_all=idx_all,
        noise_var_real=noise_var_real,
        snr_db=snr_db_arr,
        config=json.dumps(cfg.__dict__, ensure_ascii=False, indent=2,
                          default=str),
    )
    if Y_clean_c is not None:
        out["Y_clean_c"] = Y_clean_c

    np.savez_compressed(save_path, **out)
    print(f"Saved: {save_path}, m={m}, D={cfg.D}, "
          f"K={cfg.k_factor_db}dB, "
          f"SNR range=[{cfg.snr_db_min},{cfg.snr_db_max}] dB")


# =========================================================
# 7) Dataset Class
# =========================================================
class LAMPDataset(Dataset):
    """PyTorch Dataset for LAMP training/evaluation with online noise augmentation support."""
    def __init__(self,
                 npz_path: str,
                 split="train",
                 train_ratio=0.8,
                 device="cpu",
                 online_noise: bool = False,
                 snr_db_min: float = 0.0,
                 snr_db_max: float = 30.0):
        d = np.load(npz_path, allow_pickle=True)

        self.A_real = torch.from_numpy(d["A_real"]).float()
        self.X_real = torch.from_numpy(d["X_real"]).float()
        self.H_true = torch.from_numpy(d["H_true"]).to(torch.complex64)
        self.U = torch.from_numpy(d["U"]).to(torch.complex64)
        self.idx_all = d["idx_all"].astype(np.int64)
        self.m = self.idx_all.shape[0]

        self.online_noise = bool(online_noise)
        self.snr_db_min = float(snr_db_min)
        self.snr_db_max = float(snr_db_max)

        self.Y_real = torch.from_numpy(d["Y_real"]).float() if ("Y_real" in d) else None
        self.noise_var_real = (torch.from_numpy(d["noise_var_real"]).float()
                               if ("noise_var_real" in d) else None)
        self.Y_clean_c = (torch.from_numpy(d["Y_clean_c"]).to(torch.complex64)
                          if ("Y_clean_c" in d) else None)

        self.S = self.X_real.shape[0]
        cut = int(self.S * train_ratio)
        self.indices = (np.arange(0, cut) if split == "train"
                        else np.arange(cut, self.S))
        self.device = device

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, k):
        i = int(self.indices[k])

        Xgt = self.X_real[i].to(self.device)
        Ht = self.H_true[i].to(self.device)

        if not self.online_noise:
            Y = self.Y_real[i].to(self.device)
            nv = self.noise_var_real[i].to(self.device)
            return Y, Xgt, nv, Ht

        # --- Online noise augmentation ---
        if self.Y_clean_c is not None:
            y_clean = self.Y_clean_c[i].to(self.device)
        else:
            y_clean = Ht[self.idx_all]

        # Sample random SNR
        snr_db = (self.snr_db_min +
                  (self.snr_db_max - self.snr_db_min) *
                  float(torch.rand((), device=self.device)))
        snr_lin = 10.0 ** (snr_db / 10.0)

        sig_pow = torch.mean(torch.abs(y_clean) ** 2) + 1e-12
        sigma2_complex = sig_pow / snr_lin
        sigma2_real = sigma2_complex / 2.0

        n = ((torch.randn(self.m, device=self.device) +
              1j * torch.randn(self.m, device=self.device)) *
             torch.sqrt(sigma2_complex / 2.0))
        y = y_clean + n

        Y = torch.cat([y.real, y.imag], dim=0).unsqueeze(1).float()
        nv = sigma2_real.float()

        return Y, Xgt, nv, Ht


# =========================================================
# 8) LAMP Network
# =========================================================
def gaussian_mmse_denoise(r: torch.Tensor, D: int,
                          gamma: torch.Tensor,
                          lam_c: torch.Tensor
                          ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Gaussian MMSE denoiser for complex coefficients."""
    B = r.shape[0]
    r_re = r[:, :D, :]
    r_im = r[:, D:, :]

    lam_c = lam_c.clamp_min(1e-6).view(1, D, 1)
    gamma_x = 2.0 / lam_c

    a = gamma / (gamma + gamma_x)
    x_re = a * r_re
    x_im = a * r_im
    x = torch.cat([x_re, x_im], dim=1)

    div = torch.mean(a, dim=1, keepdim=True)
    div = torch.clamp(div, 1e-4, 1.0 - 1e-4)
    return x, div


class LAMPLayer(nn.Module):
    """Single LAMP layer with learned step size, Onsager scaling, and damping."""
    def __init__(self, N2: int, m2: int, D: int):
        super().__init__()
        self.N2 = N2
        self.m2 = m2
        self.D = D

        self.step = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.log_beta = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        self.logit_damp_x = nn.Parameter(torch.tensor(-0.5, dtype=torch.float32))
        self.logit_damp_v = nn.Parameter(torch.tensor(-0.5, dtype=torch.float32))

    def forward(self, x, v, A, A_T, y, noise_var_real, lam_c):
        res = y - torch.bmm(A, x)

        # Estimate effective variance
        tau2_est = torch.mean(res ** 2, dim=1, keepdim=True)
        tau2 = torch.clamp(tau2_est, min=noise_var_real.view(-1, 1, 1))
        gamma = 1.0 / (tau2 + 1e-12)

        r = x + self.step * torch.bmm(A_T, res)
        x_new, div = gaussian_mmse_denoise(r, self.D, gamma, lam_c)

        # Onsager correction
        beta = torch.exp(self.log_beta).view(1, 1, 1)
        ons = beta * (self.N2 / self.m2) * div
        v_new = y - torch.bmm(A, x_new) + ons * v

        # Damping
        damp_x = torch.sigmoid(self.logit_damp_x)
        damp_v = torch.sigmoid(self.logit_damp_v)
        x_out = damp_x * x + (1.0 - damp_x) * x_new
        v_out = damp_v * v + (1.0 - damp_v) * v_new
        return x_out, v_out


class LAMPNet(nn.Module):
    """LAMP (Learned Approximate Message Passing) network for FAS channel estimation."""
    def __init__(self, A_real: torch.Tensor, D: int, num_layers: int = 8):
        super().__init__()
        self.D = D
        self.N2 = 2 * D
        self.m2 = A_real.shape[0]

        self.register_buffer("A_real", A_real)
        self.log_lam_c = nn.Parameter(torch.zeros(D))

        # Learnable initial state to capture LoS component in Rician channels
        self.x_init = nn.Parameter(torch.zeros(self.N2, 1))

        self.layers = nn.ModuleList([
            LAMPLayer(N2=self.N2, m2=self.m2, D=D)
            for _ in range(num_layers)
        ])

    def forward(self, y: torch.Tensor, noise_var_real: torch.Tensor):
        B = y.shape[0]
        A = self.A_real.unsqueeze(0).expand(B, -1, -1)
        A_T = A.transpose(1, 2)

        # Initialize with learned mean
        x = self.x_init.unsqueeze(0).expand(B, -1, -1)
        
        # Initial residual based on learned initial state
        v = y - torch.bmm(A, x)

        lam_c = torch.exp(self.log_lam_c)

        all_x = []
        all_res = []

        res_0 = torch.mean((y - torch.bmm(A, x)) ** 2).item()
        all_res.append(res_0)

        for layer in self.layers:
            x, v = layer(x, v, A, A_T, y, noise_var_real, lam_c)
            all_x.append(x)
            curr_res = y - torch.bmm(A, x)
            mse = torch.mean(curr_res ** 2).item()
            all_res.append(mse)

        return x, all_x, all_res


# =========================================================
# 9) Evaluation Metrics
# =========================================================
@torch.no_grad()
def nmse_db_h_from_coeff(x_hat: torch.Tensor,
                         H_true: torch.Tensor,
                         U: torch.Tensor,
                         D: int) -> float:
    """Compute NMSE (dB) of channel estimate from coefficient estimate."""
    c_re = x_hat[:, :D, 0]
    c_im = x_hat[:, D:, 0]
    c = torch.complex(c_re, c_im)
    h_hat = torch.matmul(U.unsqueeze(0), c.unsqueeze(2)).squeeze(2)
    err = H_true - h_hat
    nmse = torch.mean(
        torch.sum(torch.abs(err) ** 2, dim=1) /
        (torch.sum(torch.abs(H_true) ** 2, dim=1) + 1e-12)
    )
    return float(10.0 * torch.log10(nmse + 1e-12).item())


# =========================================================
# 10) Checkpoint Utilities
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
        optimizer.load_state_dict(ckpt["optimizer"])
    epoch = int(ckpt.get("epoch", 0))
    best_val_loss = float(ckpt.get("best_val_loss", 1e9))
    return epoch, best_val_loss, ckpt


# =========================================================
# 11) Single Scenario Training Function
# =========================================================
def train_lamp_single(npz_path: str,
                      device: str,
                      total_epochs: int = 200,
                      resume_path: str = "lamp_last_ckpt.pth",
                      best_path: str = "lamp_fas_best.pth",
                      snr_db_min_train: float = 0.0,
                      snr_db_max_train: float = 30.0,
                      snr_db_val: float = 10.0,
                      scenario_tag: str = ""):
    """
    Train LAMP for a single scenario.
    Training SNR is uniformly sampled from [snr_db_min_train, snr_db_max_train].
    Validation uses fixed SNR at snr_db_val.
    """
    print(f"\n{'=' * 60}")
    print(f"Training LAMP — {scenario_tag}")
    print(f"  Dataset : {npz_path}")
    print(f"  SNR train: [{snr_db_min_train}, {snr_db_max_train}] dB")
    print(f"  SNR val  : {snr_db_val} dB")
    print(f"  Epochs   : {total_epochs}")
    print(f"  Best path: {best_path}")
    print(f"{'=' * 60}\n")

    train_ds = LAMPDataset(
        npz_path, split="train", device=device,
        online_noise=True,
        snr_db_min=snr_db_min_train,
        snr_db_max=snr_db_max_train,
    )
    val_ds = LAMPDataset(
        npz_path, split="val", device=device,
        online_noise=True,
        snr_db_min=snr_db_val,
        snr_db_max=snr_db_val,
    )
    train_loader = DataLoader(train_ds, batch_size=256,
                              shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=512, shuffle=False)

    A_real = train_ds.A_real.to(device)
    U = train_ds.U.to(device)
    D = train_ds.X_real.shape[1] // 2

    model = LAMPNet(A_real=A_real, D=D, num_layers=10).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    crit = nn.MSELoss()

    start_epoch = 1
    best_val_loss = 1e9

    # Resume from checkpoint if exists
    if resume_path is not None and os.path.exists(resume_path):
        ep, best_val_loss, _ = load_checkpoint(
            resume_path, model, opt, map_location=device)
        start_epoch = ep + 1
        print(f"[Resume] Loaded {resume_path}, "
              f"start_epoch={start_epoch}, "
              f"best_val_loss={best_val_loss:.4e}")

    for epoch in range(start_epoch, total_epochs + 1):
        # ---- Training ----
        model.train()
        tr = 0.0
        for Y, Xgt, nv, Ht in train_loader:
            x_hat, layers, _ = model(Y, nv)

            # Deep supervision
            loss = 0.0
            for j, xj in enumerate(layers):
                w = (j + 1) / len(layers)
                loss = loss + w * crit(xj, Xgt)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            tr += float(loss.item())
        tr /= max(1, len(train_loader))

        # ---- Validation ----
        model.eval()
        va = 0.0
        nmse_list, oracle_list = [], []
        with torch.no_grad():
            for Y, Xgt, nv, Ht in val_loader:
                x_hat, _, _ = model(Y, nv)
                va += float(crit(x_hat, Xgt).item())
                nmse_list.append(nmse_db_h_from_coeff(x_hat, Ht, U, D))
                oracle_list.append(nmse_db_h_from_coeff(Xgt, Ht, U, D))
        va /= max(1, len(val_loader))
        nmse = float(np.mean(nmse_list))
        oracle = float(np.mean(oracle_list))

        print(f"[{scenario_tag}] Epoch {epoch:03d} | "
              f"TrLoss={tr:.4e} | ValLoss={va:.4e} | "
              f"NMSE(h)={nmse:.2f} dB | Oracle={oracle:.2f} dB")

        # Save last checkpoint every epoch
        if resume_path is not None:
            save_checkpoint(resume_path, model, opt, epoch, best_val_loss)

        # Save best model
        if va < best_val_loss:
            best_val_loss = va
            torch.save(model.state_dict(), best_path)
            if resume_path is not None:
                save_checkpoint(resume_path, model, opt, epoch, best_val_loss)
            print(f"  -> saved best to {best_path} "
                  f"(best_val_loss={best_val_loss:.4e})")

    print(f"\n[{scenario_tag}] Training done. Best val loss = {best_val_loss:.4e}")
    return best_path


# =========================================================
# 12) Multi-Scenario Batch Training Entry Point
# =========================================================
def train_all_scenarios(D: int = 12, M: int = 8, Tpilot: int = 2,
                        S: int = 20000,
                        total_epochs: int = 200,
                        device: str = "cuda"):
    """
    Build and train all scenarios:
      1) Rayleigh (K = -inf dB)
      2) Rician K = 5 dB
      3) Rician K = 10 dB
      4) Rician K = 15 dB

    Each scenario saves separately:
      - Dataset .npz
      - LAMP last checkpoint .pth
      - LAMP best checkpoint .pth
    """
    scenarios = get_scenario_configs(D=D, M=M, Tpilot=Tpilot, S=S)

    saved_files = {}

    for name, cfg in scenarios.items():
        tag = cfg.make_tag()

        npz_path = f"fas_lamp_dataset_{tag}.npz"
        resume_path = f"lamp_last_ckpt_{tag}.pth"
        best_path = f"lamp_fas_best_{tag}.pth"

        # 1) Build dataset
        if not os.path.exists(npz_path):
            build_npz_from_fas(npz_path, cfg)
        else:
            print(f"[{name}] Dataset already exists: {npz_path}")

        # 2) Train LAMP
        train_lamp_single(
            npz_path=npz_path,
            device=device,
            total_epochs=total_epochs,
            resume_path=resume_path,
            best_path=best_path,
            snr_db_min_train=cfg.snr_db_min,
            snr_db_max_train=cfg.snr_db_max,
            snr_db_val=10.0,
            scenario_tag=name,
        )

        saved_files[name] = {
            "npz": npz_path,
            "resume": resume_path,
            "best": best_path,
            "config": cfg,
        }

    # Print summary
    print("\n" + "=" * 70)
    print("All scenarios completed!")
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
# 13) Compatibility: Legacy train_lamp Interface
# =========================================================
def train_lamp(npz_path: str,
               device: str,
               total_epochs: int = 50,
               resume_path: str = "lamp_last_ckpt.pth",
               best_path: str = "lamp_fas_best.pth"):
    """Legacy interface for backward compatibility, defaults to SNR 0~30 dB."""
    return train_lamp_single(
        npz_path=npz_path,
        device=device,
        total_epochs=total_epochs,
        resume_path=resume_path,
        best_path=best_path,
        snr_db_min_train=0.0,
        snr_db_max_train=30.0,
        snr_db_val=10.0,
        scenario_tag="default",
    )


# =========================================================
# Main Entry Point
# =========================================================
if __name__ == "__main__":
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    import argparse
    parser = argparse.ArgumentParser(description="LAMP training for FAS channel estimation")
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
    args = parser.parse_args()

    if args.mode == "all":
        # ===== Train all scenarios =====
        saved = train_all_scenarios(
            D=args.D, M=args.M, Tpilot=args.T,
            S=args.S,
            total_epochs=args.epochs,
            device=device,
        )

    else:
        # ===== Train single Rayleigh scenario only =====
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
            build_npz_from_fas(npz_path, cfg)

        train_lamp(
            npz_path=npz_path,
            device=device,
            total_epochs=args.epochs,
            resume_path=f"lamp_last_ckpt_{tag}.pth",
            best_path=f"lamp_fas_best_{tag}.pth",
        )