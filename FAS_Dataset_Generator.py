# -*- coding: utf-8 -*-
"""
FAS200_Dataset_Generator.py — Standalone dataset generation for FAS channel scenarios
Generates datasets for multiple Rician K-factor scenarios
"""
import os
import math
import json
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Dict

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
# 4) Dataset Configuration
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
    print(f"✓ Saved: {save_path}")
    print(f"  m={m}, D={cfg.D}, K={cfg.k_factor_db}dB")
    print(f"  SNR range=[{cfg.snr_db_min},{cfg.snr_db_max}] dB\n")


# =========================================================
# 7) Batch Dataset Generation
# =========================================================
def generate_all_datasets(D: int = 12, M: int = 8, Tpilot: int = 2,
                         S: int = 20000,
                         output_dir: str = ".") -> Dict[str, str]:
    """
    Generate datasets for all scenarios:
      1) Rayleigh (K = -inf dB)
      2) Rician K = 5 dB
      3) Rician K = 10 dB
      4) Rician K = 15 dB

    Args:
        D: KL basis dimension
        M: Ports sampled per slot
        Tpilot: Number of pilot slots
        S: Number of samples
        output_dir: Directory to save datasets

    Returns:
        Dictionary mapping scenario names to file paths
    """
    os.makedirs(output_dir, exist_ok=True)
    
    scenarios = get_scenario_configs(D=D, M=M, Tpilot=Tpilot, S=S)
    dataset_paths = {}

    print("\n" + "=" * 70)
    print(f"Generating FAS Channel Datasets")
    print("=" * 70)
    print(f"Configuration:")
    print(f"  D={D}, M={M}, T={Tpilot}, Samples={S}")
    print(f"  Output directory: {output_dir}")
    print("=" * 70)

    for name, cfg in scenarios.items():
        tag = cfg.make_tag()
        npz_path = os.path.join(output_dir, f"fas_lamp_dataset_{tag}.npz")

        if os.path.exists(npz_path):
            print(f"\n[{name}] Dataset already exists: {npz_path}")
            print(f"  Skipping generation. Delete file to regenerate.")
        else:
            build_npz_from_fas(npz_path, cfg)

        dataset_paths[name] = npz_path

    # Print summary
    print("\n" + "=" * 70)
    print("Dataset Generation Summary")
    print("=" * 70)
    for name, path in dataset_paths.items():
        exists = "✓" if os.path.exists(path) else "✗"
        k_db = scenarios[name].k_factor_db
        fading_type = "Rayleigh" if k_db < -50 else f"Rician K={k_db}dB"
        print(f"\n[{exists}] {name}")
        print(f"    Type: {fading_type}")
        print(f"    Path: {path}")

    print("\n" + "=" * 70)
    print("All datasets ready!")
    print("=" * 70 + "\n")

    return dataset_paths


# =========================================================
# 8) Single Dataset Generation (for custom scenarios)
# =========================================================
def generate_single_dataset(k_factor_db: float = -np.inf,
                           D: int = 12,
                           M: int = 8,
                           Tpilot: int = 2,
                           S: int = 20000,
                           snr_db_min: float = 0.0,
                           snr_db_max: float = 30.0,
                           output_dir: str = ".") -> str:
    """
    Generate a single custom dataset.

    Args:
        k_factor_db: Rician K-factor in dB (-inf for Rayleigh)
        D: KL basis dimension
        M: Ports sampled per slot
        Tpilot: Number of pilot slots
        S: Number of samples
        snr_db_min: Minimum SNR (dB)
        snr_db_max: Maximum SNR (dB)
        output_dir: Directory to save dataset

    Returns:
        Path to generated dataset
    """
    os.makedirs(output_dir, exist_ok=True)

    # Determine scenario tag
    if k_factor_db < -50:
        scenario_tag = "rayleigh"
    else:
        scenario_tag = f"rician_{k_factor_db:.0f}dB"

    cfg = GenCfg(
        S=S, N=200, W_lambda=4.0, fc_ghz=3.5,
        aoa_mean_deg=20.0, aoa_spread_deg=10.0,
        use_isotropic=False,
        k_factor_db=k_factor_db,
        D=D, M=M, Tpilot=Tpilot,
        snr_db_min=snr_db_min,
        snr_db_max=snr_db_max,
        save_y_clean=True,
        seed=0,
        scenario_tag=scenario_tag,
    )

    tag = cfg.make_tag()
    npz_path = os.path.join(output_dir, f"fas_lamp_dataset_{tag}.npz")

    build_npz_from_fas(npz_path, cfg)

    return npz_path


# =========================================================
# Main Entry Point
# =========================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate FAS channel datasets for LAMP training"
    )
    parser.add_argument("--mode", type=str, default="all",
                        choices=["all", "single", "custom"],
                        help="Generation mode: 'all' for predefined scenarios, "
                             "'single' for Rayleigh only, 'custom' for custom K-factor")
    parser.add_argument("--D", type=int, default=12,
                        help="KL basis dimension")
    parser.add_argument("--M", type=int, default=8,
                        help="Ports sampled per slot")
    parser.add_argument("--T", type=int, default=2,
                        help="Number of pilot slots")
    parser.add_argument("--S", type=int, default=20000,
                        help="Number of samples")
    parser.add_argument("--K", type=float, default=-np.inf,
                        help="Rician K-factor in dB (for custom mode)")
    parser.add_argument("--snr_min", type=float, default=0.0,
                        help="Minimum SNR in dB")
    parser.add_argument("--snr_max", type=float, default=30.0,
                        help="Maximum SNR in dB")
    parser.add_argument("--output_dir", type=str, default=".",
                        help="Output directory for datasets")

    args = parser.parse_args()

    if args.mode == "all":
        # Generate all predefined scenarios
        print("\n🚀 Generating all predefined scenarios...")
        paths = generate_all_datasets(
            D=args.D, M=args.M, Tpilot=args.T, S=args.S,
            output_dir=args.output_dir
        )

    elif args.mode == "single":
        # Generate single Rayleigh scenario
        print("\n🚀 Generating single Rayleigh scenario...")
        path = generate_single_dataset(
            k_factor_db=-np.inf,
            D=args.D, M=args.M, Tpilot=args.T, S=args.S,
            snr_db_min=args.snr_min, snr_db_max=args.snr_max,
            output_dir=args.output_dir
        )
        print(f"\n✓ Dataset saved to: {path}")

    else:  # custom
        # Generate custom K-factor scenario
        print(f"\n🚀 Generating custom scenario (K={args.K} dB)...")
        path = generate_single_dataset(
            k_factor_db=args.K,
            D=args.D, M=args.M, Tpilot=args.T, S=args.S,
            snr_db_min=args.snr_min, snr_db_max=args.snr_max,
            output_dir=args.output_dir
        )
        print(f"\n✓ Dataset saved to: {path}")# -*- coding: utf-8 -*-
"""
FAS200_Dataset_Generator.py — Standalone dataset generation for FAS channel scenarios
Generates datasets for multiple Rician K-factor scenarios
"""
import os
import math
import json
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Dict

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
# 4) Dataset Configuration
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
    print(f"✓ Saved: {save_path}")
    print(f"  m={m}, D={cfg.D}, K={cfg.k_factor_db}dB")
    print(f"  SNR range=[{cfg.snr_db_min},{cfg.snr_db_max}] dB\n")


# =========================================================
# 7) Batch Dataset Generation
# =========================================================
def generate_all_datasets(D: int = 12, M: int = 8, Tpilot: int = 2,
                         S: int = 20000,
                         output_dir: str = ".") -> Dict[str, str]:
    """
    Generate datasets for all scenarios:
      1) Rayleigh (K = -inf dB)
      2) Rician K = 5 dB
      3) Rician K = 10 dB
      4) Rician K = 15 dB

    Args:
        D: KL basis dimension
        M: Ports sampled per slot
        Tpilot: Number of pilot slots
        S: Number of samples
        output_dir: Directory to save datasets

    Returns:
        Dictionary mapping scenario names to file paths
    """
    os.makedirs(output_dir, exist_ok=True)
    
    scenarios = get_scenario_configs(D=D, M=M, Tpilot=Tpilot, S=S)
    dataset_paths = {}

    print("\n" + "=" * 70)
    print(f"Generating FAS Channel Datasets")
    print("=" * 70)
    print(f"Configuration:")
    print(f"  D={D}, M={M}, T={Tpilot}, Samples={S}")
    print(f"  Output directory: {output_dir}")
    print("=" * 70)

    for name, cfg in scenarios.items():
        tag = cfg.make_tag()
        npz_path = os.path.join(output_dir, f"fas_lamp_dataset_{tag}.npz")

        if os.path.exists(npz_path):
            print(f"\n[{name}] Dataset already exists: {npz_path}")
            print(f"  Skipping generation. Delete file to regenerate.")
        else:
            build_npz_from_fas(npz_path, cfg)

        dataset_paths[name] = npz_path

    # Print summary
    print("\n" + "=" * 70)
    print("Dataset Generation Summary")
    print("=" * 70)
    for name, path in dataset_paths.items():
        exists = "✓" if os.path.exists(path) else "✗"
        k_db = scenarios[name].k_factor_db
        fading_type = "Rayleigh" if k_db < -50 else f"Rician K={k_db}dB"
        print(f"\n[{exists}] {name}")
        print(f"    Type: {fading_type}")
        print(f"    Path: {path}")

    print("\n" + "=" * 70)
    print("All datasets ready!")
    print("=" * 70 + "\n")

    return dataset_paths


# =========================================================
# 8) Single Dataset Generation (for custom scenarios)
# =========================================================
def generate_single_dataset(k_factor_db: float = -np.inf,
                           D: int = 12,
                           M: int = 8,
                           Tpilot: int = 2,
                           S: int = 20000,
                           snr_db_min: float = 0.0,
                           snr_db_max: float = 30.0,
                           output_dir: str = ".") -> str:
    """
    Generate a single custom dataset.

    Args:
        k_factor_db: Rician K-factor in dB (-inf for Rayleigh)
        D: KL basis dimension
        M: Ports sampled per slot
        Tpilot: Number of pilot slots
        S: Number of samples
        snr_db_min: Minimum SNR (dB)
        snr_db_max: Maximum SNR (dB)
        output_dir: Directory to save dataset

    Returns:
        Path to generated dataset
    """
    os.makedirs(output_dir, exist_ok=True)

    # Determine scenario tag
    if k_factor_db < -50:
        scenario_tag = "rayleigh"
    else:
        scenario_tag = f"rician_{k_factor_db:.0f}dB"

    cfg = GenCfg(
        S=S, N=200, W_lambda=4.0, fc_ghz=3.5,
        aoa_mean_deg=20.0, aoa_spread_deg=10.0,
        use_isotropic=False,
        k_factor_db=k_factor_db,
        D=D, M=M, Tpilot=Tpilot,
        snr_db_min=snr_db_min,
        snr_db_max=snr_db_max,
        save_y_clean=True,
        seed=0,
        scenario_tag=scenario_tag,
    )

    tag = cfg.make_tag()
    npz_path = os.path.join(output_dir, f"fas_lamp_dataset_{tag}.npz")

    build_npz_from_fas(npz_path, cfg)

    return npz_path


# =========================================================
# Main Entry Point
# =========================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate FAS channel datasets for LAMP training"
    )
    parser.add_argument("--mode", type=str, default="all",
                        choices=["all", "single", "custom"],
                        help="Generation mode: 'all' for predefined scenarios, "
                             "'single' for Rayleigh only, 'custom' for custom K-factor")
    parser.add_argument("--D", type=int, default=12,
                        help="KL basis dimension")
    parser.add_argument("--M", type=int, default=8,
                        help="Ports sampled per slot")
    parser.add_argument("--T", type=int, default=2,
                        help="Number of pilot slots")
    parser.add_argument("--S", type=int, default=20000,
                        help="Number of samples")
    parser.add_argument("--K", type=float, default=-np.inf,
                        help="Rician K-factor in dB (for custom mode)")
    parser.add_argument("--snr_min", type=float, default=0.0,
                        help="Minimum SNR in dB")
    parser.add_argument("--snr_max", type=float, default=30.0,
                        help="Maximum SNR in dB")
    parser.add_argument("--output_dir", type=str, default=".",
                        help="Output directory for datasets")

    args = parser.parse_args()

    if args.mode == "all":
        # Generate all predefined scenarios
        print("\n🚀 Generating all predefined scenarios...")
        paths = generate_all_datasets(
            D=args.D, M=args.M, Tpilot=args.T, S=args.S,
            output_dir=args.output_dir
        )

    elif args.mode == "single":
        # Generate single Rayleigh scenario
        print("\n🚀 Generating single Rayleigh scenario...")
        path = generate_single_dataset(
            k_factor_db=-np.inf,
            D=args.D, M=args.M, Tpilot=args.T, S=args.S,
            snr_db_min=args.snr_min, snr_db_max=args.snr_max,
            output_dir=args.output_dir
        )
        print(f"\n✓ Dataset saved to: {path}")

    else:  # custom
        # Generate custom K-factor scenario
        print(f"\n🚀 Generating custom scenario (K={args.K} dB)...")
        path = generate_single_dataset(
            k_factor_db=args.K,
            D=args.D, M=args.M, Tpilot=args.T, S=args.S,
            snr_db_min=args.snr_min, snr_db_max=args.snr_max,
            output_dir=args.output_dir
        )
        print(f"\n✓ Dataset saved to: {path}")