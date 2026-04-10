# eval_rician_scenarios.py
# -*- coding: utf-8 -*-
"""
Evaluate NMSE under different Rician K-factor scenarios
for all methods: OMP, LMMSE(Scalar prior), LMMSE(Diagonal prior), GAMP, FCNet, LAMP, Genie-LMMSE(true AoA)
* WITH Rician LoS Separation Strategy *
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib import rcParams
from typing import Dict, List, Tuple
from dataclasses import dataclass

from FAS_KLLAMP import LAMPNet
from FAS_FCnet import FCNetEnhanced

# Set font to Times New Roman
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times New Roman']
rcParams['mathtext.fontset'] = 'stix'
rcParams['font.size'] = 11
rcParams['axes.grid'] = True
rcParams['grid.alpha'] = 0.3


def load_fcnet(ckpt_path: str, in_dim: int, out_dim: int, device: torch.device):
    """Load FCNet model from checkpoint."""
    if not os.path.exists(ckpt_path):
        print(f"  [FCNet] WARNING: {ckpt_path} not found!")
        return FCNetEnhanced(in_dim, out_dim).to(device)
    state = torch.load(ckpt_path, map_location=device)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    is_enhanced = any("input_proj" in k or "res_blocks" in k for k in state.keys())
    model = FCNetEnhanced(in_dim, out_dim).to(device) if is_enhanced else FCNet(in_dim, out_dim).to(device)
    model.load_state_dict(state)
    print(f"  [FCNet] Loaded {'Enhanced' if is_enhanced else 'Basic'} from {ckpt_path}")
    return model


# =========================================================
# Helper Functions
# =========================================================
def nmse_num_den(h_hat, h_true):
    """Compute numerator and denominator for NMSE calculation."""
    num = np.vdot(h_true - h_hat, h_true - h_hat).real
    den = np.vdot(h_true, h_true).real + 1e-12
    return float(num), float(den)


def omp_complex(y, A, L):
    """Orthogonal Matching Pursuit for complex signals."""
    m, K = A.shape
    r, support = y.copy(), []
    x_hat = np.zeros(K, dtype=complex)
    for _ in range(L):
        idx = int(np.argmax(np.abs(A.conj().T @ r)))
        if idx not in support:
            support.append(idx)
        As = A[:, support]
        xs, *_ = np.linalg.lstsq(As, y, rcond=None)
        r = y - As @ xs
    x_hat[support] = xs
    return x_hat


def lmmse_real_scalar(y, A, sigma2, tau_x=1.0):
    """Robust LMMSE with scalar prior (isotropic)."""
    M, N = A.shape
    lam = sigma2 / (tau_x + 1e-12)
    W = np.linalg.inv(A.T @ A + lam * np.eye(N))
    return W @ (A.T @ y)


def lmmse_real_diag(y, A, sigma2, lam_c):
    """
    LMMSE with diagonal prior.
    Prior: c ~ CN(0, diag(lam_c)), where lam_c[k] = E|c_k|^2 (complex variance).
    In real domain x=[Re(c);Im(c)], diag prior becomes diag([lam_c/2, lam_c/2]).
    """
    M2, N2 = A.shape
    D = N2 // 2
    assert lam_c.shape[0] == D

    diag_vals = np.concatenate([lam_c, lam_c]) / 2.0
    diag_vals = np.maximum(diag_vals, 1e-12)

    G = A.T @ A
    Inv_R = np.diag(sigma2 / diag_vals)
    W = np.linalg.inv(G + Inv_R)
    return W @ (A.T @ y)


def gaussian_mmse_denoise_np(r, D, gamma, lam_c):
    """Gaussian MMSE denoiser in numpy."""
    r = r.reshape(-1)
    lam_c = np.maximum(lam_c, 1e-6)
    a = gamma / (gamma + 2.0 / lam_c)
    x_new = np.concatenate([a * r[:D], a * r[D:]])[:, None]
    return x_new, float(np.mean(a))


def amp_gaussian_real_baseline(y_real, A_real, sigma2_real, lam_c, D, num_iter=20):
    """AMP with Gaussian diagonal prior."""
    y = y_real.reshape(-1, 1).astype(np.float64)
    A = A_real.astype(np.float64)
    m2, N2 = A.shape
    x, v = np.zeros((N2, 1)), y.copy()
    
    for _ in range(num_iter):
        res = y - A @ x
        tau2 = max(float(np.mean(res ** 2)), sigma2_real)
        gamma = 1.0 / (tau2 + 1e-12)
        r = x + A.T @ res
        x_new, div = gaussian_mmse_denoise_np(r, D, gamma, lam_c)
        v = y - A @ x_new + (N2 / m2) * div * v
        x = x_new
    return x.reshape(-1)


# =========================================================
# Scenario File Path Management
# =========================================================
@dataclass
class ScenarioFiles:
    """Data class for scenario file paths."""
    name: str
    k_factor_db: float
    npz_path: str
    lamp_ckpt: str
    fcnet_ckpt: str

    @property
    def display_name(self):
        return "Rayleigh" if self.k_factor_db < -50 else f"Rician $K={self.k_factor_db:.0f}$ dB"


def get_scenario_file_paths(D=12, M=8, Tpilot=2, base_dir="."):
    """Generate file paths for all scenarios."""
    scenarios = {}
    scenario_defs = [("rayleigh", -np.inf), ("rician_5dB", 5.0), ("rician_10dB", 10.0), ("rician_15dB", 15.0)]
    shared_lamp = os.path.join(base_dir, f"lamp_fas_best_D{D}_M{M}_T{Tpilot}_rayleigh_snr0to30.pth")
    shared_fcnet = os.path.join(base_dir, f"fcnet_fas_best_D{D}_M{M}_T{Tpilot}_rayleigh_snr0to30.pth")

    for name, k_db in scenario_defs:
        tag = f"D{D}_M{M}_T{Tpilot}_rayleigh_snr0to30" if k_db < -50 else f"D{D}_M{M}_T{Tpilot}_K{k_db:.0f}dB_snr0to30"
        scenarios[name] = ScenarioFiles(
            name=name, k_factor_db=k_db,
            npz_path=os.path.join(base_dir, f"fas_lamp_dataset_{tag}.npz"),
            lamp_ckpt=shared_lamp, fcnet_ckpt=shared_fcnet)
    return scenarios


# =========================================================
# Core Evaluation Function
# =========================================================
@torch.no_grad()
def evaluate_single_scenario(npz_path, lamp_ckpt, fcnet_ckpt, device,
                             k_factor_db, snr_db_list, max_samples=2000, omp_sparsity=10, num_layers=10):
    """Evaluate all methods on a single scenario."""
    d = np.load(npz_path, allow_pickle=True)
    A_real = d["A_real"].astype(np.float64)
    H_true = d["H_true"].astype(np.complex128)
    U = d["U"].astype(np.complex128)
    idx_all = d["idx_all"].astype(np.int64)

    S, N = H_true.shape
    m = A_real.shape[0] // 2
    D = A_real.shape[1] // 2
    T = min(max_samples, S)

    # Compute prior parameters (for LMMSE Scalar/Diagonal and GAMP)
    Xr = d["X_real"][:, :, 0].astype(np.float64)
    var_dims = np.var(Xr, axis=0)
    lam_c = var_dims[:D] + var_dims[D:]  # Diagonal prior (nominal)
    tau_x = float(np.mean(var_dims))     # Scalar prior
    A_c = U[idx_all, :]

    # Load models
    A_real_torch = torch.from_numpy(A_real).float().to(device)
    lamp = LAMPNet(A_real=A_real_torch, D=D, num_layers=num_layers).to(device)
    if os.path.exists(lamp_ckpt):
        state = torch.load(lamp_ckpt, map_location=device)
        lamp.load_state_dict(state.get("model", state))
    lamp.eval()

    fcnet = load_fcnet(fcnet_ckpt, 2 * m, 2 * D, device)
    fcnet.eval()

    rng = np.random.default_rng(0)
    methods = ["OMP", "LMMSE(Scalar prior)", "LMMSE(Diagonal prior)", "GAMP", "FCNet", "LAMP", "Genie-LMMSE(true AoA)"]
    nmse_db_results = {k: [] for k in methods}

    # LoS separation preparation
    is_rician = (k_factor_db > -50)
    x_positions = np.linspace(0, 4.0, N)
    h_LoS_full = np.exp(1j * 2 * np.pi * x_positions * np.sin(np.deg2rad(20.0)))
    y_LoS_clean = h_LoS_full[idx_all]
    w_LoS = np.sqrt(10 ** (k_factor_db / 10.0) / (10 ** (k_factor_db / 10.0) + 1)) if is_rician else 0.0

    for snr_db in snr_db_list:
        acc_num = {k: 0.0 for k in methods}
        acc_den = {k: 0.0 for k in methods}
        Y_real_batch = np.zeros((T, 2 * m))
        sigma2_r_batch = np.zeros(T)

        for s_idx in range(T):
            h_actual = H_true[s_idx]
            y_clean_actual = h_actual[idx_all]
            sig_pow = np.mean(np.abs(y_clean_actual) ** 2) + 1e-12
            snr_lin = 10 ** (snr_db / 10.0)
            sigma2_c = sig_pow / snr_lin
            sigma2_r = sigma2_c / 2.0

            noise = (rng.standard_normal(m) + 1j * rng.standard_normal(m)) * np.sqrt(sigma2_c / 2.0)
            y_actual = y_clean_actual + noise
            y_resid = y_actual - w_LoS * y_LoS_clean if is_rician else y_actual

            y_real = np.concatenate([y_resid.real, y_resid.imag])
            Y_real_batch[s_idx] = y_real
            sigma2_r_batch[s_idx] = sigma2_r

            def reconstruct(c_hat):
                h_res = U @ c_hat
                return h_res + w_LoS * h_LoS_full if is_rician else h_res

            # OMP
            c_omp = omp_complex(y_resid, A_c, min(omp_sparsity, D))
            num, den = nmse_num_den(reconstruct(c_omp), h_actual)
            acc_num["OMP"] += num; acc_den["OMP"] += den

            # LMMSE (Scalar prior) - Robust
            x_scalar = lmmse_real_scalar(y_real, A_real, sigma2_r, tau_x)
            num, den = nmse_num_den(reconstruct(x_scalar[:D] + 1j * x_scalar[D:]), h_actual)
            acc_num["LMMSE(Scalar prior)"] += num; acc_den["LMMSE(Scalar prior)"] += den

            # LMMSE (Diagonal prior) - Nominal
            x_diag = lmmse_real_diag(y_real, A_real, sigma2_r, lam_c)
            num, den = nmse_num_den(reconstruct(x_diag[:D] + 1j * x_diag[D:]), h_actual)
            acc_num["LMMSE(Diagonal prior)"] += num; acc_den["LMMSE(Diagonal prior)"] += den

            # GAMP
            x_amp = amp_gaussian_real_baseline(y_real, A_real, sigma2_r, lam_c, D)
            num, den = nmse_num_den(reconstruct(x_amp[:D] + 1j * x_amp[D:]), h_actual)
            acc_num["GAMP"] += num; acc_den["GAMP"] += den

            # Genie-LMMSE(true AoA) - Use true diagonal prior (matched to current K-factor)
            x_genie = lmmse_real_diag(y_real, A_real, sigma2_r, lam_c)
            num, den = nmse_num_den(reconstruct(x_genie[:D] + 1j * x_genie[D:]), h_actual)
            acc_num["Genie-LMMSE(true AoA)"] += num; acc_den["Genie-LMMSE(true AoA)"] += den

        # Batch processing for DL methods
        Y_t = torch.from_numpy(Y_real_batch).float().to(device).unsqueeze(2)
        nv_t = torch.from_numpy(sigma2_r_batch).float().to(device)

        # FCNet
        x_fc = fcnet(Y_t).squeeze(-1).cpu().numpy()
        H_fc = (U @ (x_fc[:, :D] + 1j * x_fc[:, D:]).T).T
        for s_idx in range(T):
            h_hat = H_fc[s_idx] + w_LoS * h_LoS_full if is_rician else H_fc[s_idx]
            num, den = nmse_num_den(h_hat, H_true[s_idx])
            acc_num["FCNet"] += num; acc_den["FCNet"] += den

        # LAMP
        x_lamp, *_ = lamp(Y_t, nv_t)
        x_lamp = x_lamp.squeeze(-1).cpu().numpy()
        H_lamp = (U @ (x_lamp[:, :D] + 1j * x_lamp[:, D:]).T).T
        for s_idx in range(T):
            h_hat = H_lamp[s_idx] + w_LoS * h_LoS_full if is_rician else H_lamp[s_idx]
            num, den = nmse_num_den(h_hat, H_true[s_idx])
            acc_num["LAMP"] += num; acc_den["LAMP"] += den

        for k in methods:
            nmse_db_results[k].append(10 * np.log10(max(1e-12, acc_num[k] / max(1e-12, acc_den[k]))))

    return nmse_db_results


def evaluate_all_scenarios(scenario_files, device, snr_db_list, max_samples=2000, omp_sparsity=10):
    """Evaluate all scenarios and return results dictionary."""
    all_results = {}
    for name, sf in scenario_files.items():
        print(f"\n{'='*60}\nEvaluating: {sf.display_name}\n{'='*60}")
        if not os.path.exists(sf.npz_path):
            print(f"  WARNING: {sf.npz_path} not found")
            continue
        results = evaluate_single_scenario(
            sf.npz_path, sf.lamp_ckpt, sf.fcnet_ckpt, device,
            sf.k_factor_db, snr_db_list, max_samples, omp_sparsity)
        all_results[name] = results
        print(f"  SNR | " + " | ".join([f"{m:>7s}" for m in results.keys()]))
        for i, snr in enumerate(snr_db_list):
            print(f"  {snr:>3.0f} | " + " | ".join([f"{results[m][i]:>7.2f}" for m in results.keys()]))
    return all_results


# =========================================================
# Plotting: 2x2 Combined Figure with Titles
# =========================================================
def plot_four_scenarios_combined(all_results, snr_db_list, scenario_files, save_prefix="fig_nmse_combined"):
    """
    Generate a single 2x2 combined figure for all four Rician K-factor scenarios.
    Each subplot includes a title indicating the scenario.
    """
    
    # Define plotting properties
    colors = {
        "OMP": "tab:blue",
        "LMMSE(Scalar prior)": "tab:green",
        "LMMSE(Diagonal prior)": "tab:olive",
        "GAMP": "tab:orange",
        "FCNet": "tab:purple",
        "LAMP": "tab:red",
        "Genie-LMMSE(true AoA)": "black",
    }
    markers = {
        "OMP": "s",
        "LMMSE(Scalar prior)": "D",
        "LMMSE(Diagonal prior)": "d",
        "GAMP": "X",
        "FCNet": "v",
        "LAMP": "P",
        "Genie-LMMSE(true AoA)": "*",
    }
    linestyles = {
        "OMP": "--",
        "LMMSE(Scalar prior)": "--",
        "LMMSE(Diagonal prior)": "--",
        "GAMP": "--",
        "FCNet": "-.",
        "LAMP": "-",
        "Genie-LMMSE(true AoA)": "-",
    }

    methods = [
        "OMP", "LMMSE(Scalar prior)", "LMMSE(Diagonal prior)",
        "GAMP", "FCNet", "LAMP", "Genie-LMMSE(true AoA)"
    ]

    # Scenario order and titles
    scenario_order = ["rayleigh", "rician_5dB", "rician_10dB", "rician_15dB"]
    titles = {
        "rayleigh": "(a) Rayleigh Fading",
        "rician_5dB": "(b) Rician Fading (K = 5 dB)",
        "rician_10dB": "(c) Rician Fading (K = 10 dB)",
        "rician_15dB": "(d) Rician Fading (K = 15 dB)"
    }

    # Create 2x2 figure
    fig, axes = plt.subplots(2, 2, figsize=(8, 6), sharex=True, sharey=True)
    axes = axes.flatten()  # Flatten for easy indexing

    # Plot each scenario
    for idx, scenario_name in enumerate(scenario_order):
        ax = axes[idx]
        
        if scenario_name not in all_results:
            ax.text(0.5, 0.5, "Data Missing", ha='center', va='center', transform=ax.transAxes)
            ax.set_title(titles[scenario_name], fontsize=11, fontweight='normal')
            continue
            
        results = all_results[scenario_name]
        
        for method in methods:
            if method in results:
                # Simplify label for legend
                label = "Genie LMMSE" if method == "Genie-LMMSE(true AoA)" else method
                ax.plot(
                    snr_db_list, results[method],
                    marker=markers[method],
                    color=colors[method],
                    linestyle=linestyles[method],
                    label=label,
                    markersize=5,
                    markeredgewidth=0.8,
                    linewidth=1.5
                )

        # Set subplot title
        ax.set_title(titles[scenario_name], fontsize=11, fontweight='normal')
        
        # Set axis properties
        ax.set_xlim([snr_db_list[0], snr_db_list[-1]])
        ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.4)
        ax.tick_params(direction='in', length=3, width=0.6)
        
        # Only add x-label for bottom row
        if idx >= 2:
            ax.set_xlabel("SNR (dB)", fontsize=11)
        
        # Only add y-label for left column
        if idx % 2 == 0:
            ax.set_ylabel("NMSE (dB)", fontsize=11)

    # Create shared legend at the bottom
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc='lower center',
        bbox_to_anchor=(0.5, -0.02),
        ncol=4,
        fontsize=9,
        frameon=False,
        handlelength=2.0,
        columnspacing=1.0
    )

    # Add main title
    fig.suptitle("NMSE Performance Comparison under Different Fading Scenarios", 
                 fontsize=12, fontweight='bold', y=0.98)

    # Adjust layout
    fig.tight_layout(rect=[0, 0.08, 1, 0.95])
    
    # Save in multiple formats
    fig.savefig(f"{save_prefix}.png", dpi=300, bbox_inches='tight', pad_inches=0.05)
        
    print(f"  [SAVED] {save_prefix}.png")
    plt.show()
    plt.close(fig)


# =========================================================
# Main Program
# =========================================================
if __name__ == "__main__":
    import argparse
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    parser = argparse.ArgumentParser(description="Evaluate NMSE under different Rician K-factor scenarios")
    parser.add_argument("--D", type=int, default=12, help="KL basis dimension")
    parser.add_argument("--M", type=int, default=8, help="Number of sampled ports per slot")
    parser.add_argument("--T", type=int, default=2, help="Number of pilot slots")
    parser.add_argument("--max_samples", type=int, default=2000, help="Maximum samples for evaluation")
    parser.add_argument("--omp_sparsity", type=int, default=10, help="OMP sparsity level")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    files = get_scenario_file_paths(D=args.D, M=args.M, Tpilot=args.T)

    print("\nChecking files:")
    for name, sf in files.items():
        print(f"  [{'OK' if os.path.exists(sf.npz_path) else 'MISSING'}] {sf.display_name}")

    snr_db_list = [-10, -5, 0, 5, 10, 15, 20, 25, 30]

    results = evaluate_all_scenarios(files, device, snr_db_list, args.max_samples, args.omp_sparsity)

    if results:
        plot_four_scenarios_combined(results, snr_db_list, files)
        print("\nEvaluation complete!")