import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import cholesky

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12

class FASChannelWithAoA:
    """
    Generate realistic FAS channel realizations with:
      - Large-scale fading (path loss + shadowing)
      - Small-scale fading with spatial correlation
      - Directional AoA modeling and Rician K-factor support

    Parameters
    ----------
    num_ports : int
        Number of ports on the FAS.
    fas_length_lambda : float
        Total length of FAS in wavelengths (λ).
    fc_ghz : float
        Carrier frequency in GHz.
    min_dist, max_dist : float
        Min/max user distance from BS (meters).
    use_isotropic : bool
        If True, use isotropic J0 model; else use directional Gaussian-AoA model.
    aoa_mean_deg : float
        Mean angle of arrival (degrees from normal).
    aoa_spread_deg : float
        Angular spread standard deviation (degrees).
    k_factor_db : float or None
        Rician K factor in dB. Use None or negative value for Rayleigh-only.
    """
    def __init__(self,
                 num_ports=200,
                 fas_length_lambda=2.0,
                 fc_ghz=3.5,
                 min_dist=50, max_dist=500,
                 # AoA / LoS control
                 use_isotropic=False,          # If True, use isotropic J0 model; if False, use directional Gaussian AoA
                 aoa_mean_deg=20.0,            # Mean AoA relative to array normal (degrees)
                 aoa_spread_deg=10.0,          # Angular spread standard deviation (degrees)
                 k_factor_db=5.0,              # Rician K (dB); 0 is equivalent to Rayleigh; <0 approaches Rayleigh
                 fixed_los_phase=False          # Whether the LoS phase is fixed (False = random each time)
                 ):
        self.N = num_ports
        self.W = fas_length_lambda
        self.fc = fc_ghz
        self.min_dist = min_dist
        self.max_dist = max_dist

        self.use_isotropic = use_isotropic
        self.aoa_mean = np.deg2rad(aoa_mean_deg)
        self.aoa_spread = np.deg2rad(aoa_spread_deg)
        self.K_lin = self._db_to_linear(k_factor_db)

        # Port positions in units of λ
        self.x = np.linspace(0, self.W, self.N)  # N positions, unit: λ
        self.k = 2*np.pi  # k = 2π/λ = 2π

        # Pre-compute Cholesky of the small-scale correlation matrix
        self.L_scatt = self._build_small_scale_cholesky()
        self.fixed_los_phase = fixed_los_phase  # False by default

    def _db_to_linear(self, k_db):
        """Convert K-factor in dB to linear; handle edge cases."""
        if k_db is None or np.isneginf(k_db) or k_db < -100:
            return 0.0
        return 10 ** (k_db / 10)

    # --- Large-scale fading ---
    def _large_scale(self, num_samples):
        d = np.random.uniform(self.min_dist, self.max_dist, num_samples)
        # Free-space reference loss at 1 m
        PL_d0 = 20*np.log10(4*np.pi*1*(self.fc*1e9)/3e8)
        n = 3.0
        PL = PL_d0 + 10*n*np.log10(d)
        sigma_sh = 8.0
        SH = np.random.normal(0, sigma_sh, num_samples)
        gain_db = -PL + SH
        beta = 10**(gain_db/10)
        return beta, d

    # --- Build small-scale correlation matrix ---
    def _build_small_scale_cholesky(self):
        # Isotropic: R(Δx) = J0(k Δx), where k=2π and Δx is in units of λ
        if self.use_isotropic:
            from scipy.special import j0
            X = self.x[:, None]
            dmat = np.abs(X - X.T)
            R = j0(self.k * dmat)
        else:
            # Directional Gaussian AoA approximation
            X = self.x[:, None]
            dmat = (X - X.T)  # Signed difference to preserve phase information
            phase = np.exp(1j * self.k * dmat * np.sin(self.aoa_mean))
            decay = np.exp(-0.5 * (self.k * np.abs(dmat) * np.cos(self.aoa_mean) * self.aoa_spread) ** 2)
            R = phase * decay
            # Diagonal should be 1 for numerical stability
            np.fill_diagonal(R, 1.0 + 0j)

        # Enforce Hermitian symmetry for numerical stability
        R = (R + R.conj().T) / 2
        jitter = 1e-9
        return cholesky(R + jitter*np.eye(self.N), lower=True)

    def _steering(self, theta_rad):
        # Array steering vector: a(θ)[n] = exp(j k x_n sin θ), |a_n| = 1
        return np.exp(1j * self.k * self.x * np.sin(theta_rad))
    
    def _steering_batch(self, theta_rads):
        # Batch version of the steering vector; input theta_rads: shape (S,)
        sin_thetas = np.sin(theta_rads)[:, None]  # (S, 1)
        positions = self.x[None, :]               # (1, N)
        return np.exp(1j * self.k * positions * sin_thetas)  # (S, N)

    def generate(self, num_samples, return_small=False):
        # 1) Large-scale fading
        beta, dists = self._large_scale(num_samples)

        # 2) Small-scale fading (Rayleigh scattering + optional LoS)
        # 2.1 Scattering component (correlated Rayleigh)
        z = (np.random.randn(self.N, num_samples) + 1j*np.random.randn(self.N, num_samples)) / np.sqrt(2)
        h_scatt = (self.L_scatt @ z).T  # (S, N)

        # 2.2 LoS component (each sample can have a different θ_LoS sampled around the mean AoA)
        if self.K_lin > 1e-8:
            if self.fixed_los_phase:
                # All samples share the same fixed LoS direction
                theta_los_fixed = self.aoa_mean
                A = np.tile(self._steering(theta_los_fixed), (num_samples, 1))
            else:
                theta_los = np.random.normal(self.aoa_mean, self.aoa_spread, num_samples)
                A = self._steering_batch(theta_los)
            # Power normalization: both scattering and LoS components have average per-port power ~1
            h_small = np.sqrt(self.K_lin/(1+self.K_lin)) * A + np.sqrt(1/(1+self.K_lin)) * h_scatt
        else:
            h_small = h_scatt

        # 3) Combine large-scale: h = sqrt(beta) * h_small
        h_total = h_small * np.sqrt(beta)[:, None]

        if return_small:
            return h_total, h_small, beta, dists
        return h_total, beta, dists

    def save_fas_dataset(path_npz, *, H_total, H_small, beta, dists, x, config, compress=True):
        # Unify dtype to reduce file size
        H_total = H_total.astype(np.complex64)
        H_small = H_small.astype(np.complex64)
        beta = beta.astype(np.float32)
        dists = dists.astype(np.float32)
        x = x.astype(np.float32)

        config_json = json.dumps(config, ensure_ascii=False, indent=2)

        if compress:
            np.savez_compressed(
                path_npz,
                H_total=H_total,
                H_small=H_small,
                beta=beta,
                dists=dists,
                x=x,
                config=config_json
            )
        else:
            np.savez(
                path_npz,
                H_total=H_total,
                H_small=H_small,
                beta=beta,
                dists=dists,
                x=x,
                config=config_json
            )

    def load_fas_dataset(path_npz):
        data = np.load(path_npz, allow_pickle=True)
        config = json.loads(str(data["config"]))
        return {
            "H_total": data["H_total"],
            "H_small": data["H_small"],
            "beta": data["beta"],
            "dists": data["dists"],
            "x": data["x"],
            "config": config
        }

    def visualize(self, h, beta, dists, sample_idx=0):
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12,5))
        # Left: port-wise magnitude
        plt.subplot(1,2,1)
        plt.plot(self.x, np.abs(h[sample_idx,:]), 'b-o', markersize=3 ,lw=2)
        plt.title(f'User {sample_idx} |h| across FAS\nDist={dists[sample_idx]:.1f} m')
        plt.xlabel('Position (λ)'); plt.ylabel('|h|'); plt.grid(True)
        # Right: large-scale fading histogram
        plt.subplot(1,2,2)
        plt.hist(10*np.log10(beta), bins=50, color='green', alpha=0.7)
        plt.title('Large-scale Fading (Path Loss + Shadowing)')
        plt.xlabel('Channel Gain (dB)'); plt.ylabel('Count'); plt.grid(True)
        plt.tight_layout(); plt.show()

    def visualize_all(self, H, beta, dists, indices=[0,1,2]):
        plt.figure(figsize=(12, 8))
        for i, idx in enumerate(indices):
            plt.subplot(len(indices), 1, i+1)
            mag = np.abs(H[idx])
            plt.plot(self.x, mag, '.-', label=f'Sample {idx}, Dist={dists[idx]:.1f}m')
            plt.ylabel('|h|')
            plt.legend(); plt.grid(True)
            if i == len(indices)-1:
                plt.xlabel('Position (λ)')
        plt.suptitle('FAS Channel Magnitude Profiles')
        plt.tight_layout()
        plt.show()

    def save_dataset(self, path, H, beta, dists):
        import json
        config = {
            'num_ports': self.N,
            'fas_length_lambda': self.W,
            'fc_ghz': self.fc,
            'min_dist': self.min_dist,
            'max_dist': self.max_dist,
            'use_isotropic': self.use_isotropic,
            'aoa_mean_deg': np.rad2deg(self.aoa_mean),
            'aoa_spread_deg': np.rad2deg(self.aoa_spread),
            'k_factor_db': 10*np.log10(self.K_lin) if self.K_lin > 0 else -np.inf
        }
        np.savez(path, H=H, beta=beta, dists=dists, config=np.string_(json.dumps(config, indent=2)))


import numpy as np
import matplotlib.pyplot as plt
from scipy.special import j0

# ---- Utility: estimate empirical correlation matrix from samples, with diagonal normalization ----
def empirical_R(H, remove_mean=True):
    # H: S×N
    if remove_mean:
        H = H - H.mean(axis=0, keepdims=True)  # Remove mean to eliminate LoS bias
    R = (H.conj().T @ H) / H.shape[0]
    d = np.sqrt(np.real(np.diag(R)) + 1e-12)
    R = (R / d[:,None]) / d[None,:]           # Normalize diagonal to 1
    return R

# ---- Theoretical / approximate correlation matrices (consistent with the class construction) ----
def R_theory_isotropic(x, k=2*np.pi):
    # Pure scattering, isotropic J0
    X = x[:, None]
    d = np.abs(X - X.T)
    return j0(k * d)

def R_theory_directional(x, theta0, sigma, k=2*np.pi):
    # Directional Gaussian AoA approximation
    X = x[:, None]
    d_signed = X - X.T
    phase = np.exp(1j * k * d_signed * np.sin(theta0))
    decay = np.exp(-0.5 * (k * np.abs(d_signed) * np.cos(theta0) * sigma) ** 2)
    R = phase * decay
    R = (R + R.conj().T) / 2
    np.fill_diagonal(R, 1.0 + 0j)
    return R

def R_total_with_los(x, K_lin, theta0=0.0, sigma=0.0, isotropic=True, k=2*np.pi):
    # Scattering correlation
    if isotropic:
        from scipy.special import j0
        X = x[:,None]; d = np.abs(X - X.T)
        R_scatt = j0(k*d)
    else:
        X = x[:,None]; d_signed = X - X.T
        phase = np.exp(1j*k*d_signed*np.sin(theta0))
        decay = np.exp(-0.5*(k*np.abs(d_signed)*np.cos(theta0)*sigma)**2)
        R_scatt = (phase*decay + (phase*decay).conj().T)/2
        np.fill_diagonal(R_scatt, 1.0+0j)

    # Array steering vector
    a = np.exp(1j*k*x*np.sin(theta0))
    R_los = np.outer(a, a.conj())

    # Combined correlation
    R_tot = (K_lin/(1+K_lin))*R_los + (1/(1+K_lin))*R_scatt
    # Diagonal normalization (consistent with empirical R)
    d = np.sqrt(np.real(np.diag(R_tot)) + 1e-12)
    R_tot = (R_tot / d[:,None]) / d[None,:]
    return R_tot

def empirical_r_diagonal_avg(H, remove_mean=True):
    if remove_mean:
        H = H - H.mean(axis=0, keepdims=True)
    R = (H.conj().T @ H) / H.shape[0]
    d = np.sqrt(np.real(np.diag(R)) + 1e-12)
    R = (R / d[:,None]) / d[None,:]

    N = R.shape[0]
    r = np.array([np.mean(np.diag(R, k=m)) for m in range(N)], dtype=complex)
    return r

# ---- Main verification flow ----
if __name__ == "__main__":
    # A. Build two generators: isotropic & directional AoA, LoS disabled (k_factor_db=-inf)
    # Pure scattering
    gen_iso = FASChannelWithAoA(
        num_ports=200, fas_length_lambda=4.0, fc_ghz=3.5,
        min_dist=50, max_dist=500,
        use_isotropic=True, aoa_mean_deg=0.0, aoa_spread_deg=0.0,  # Not used for isotropic
        k_factor_db=-np.inf
    )
    # Directional AoA
    gen_dir = FASChannelWithAoA(
        num_ports=200, fas_length_lambda=4.0, fc_ghz=3.5,
        min_dist=50, max_dist=500,
        use_isotropic=False, aoa_mean_deg=20.0, aoa_spread_deg=10.0,
        k_factor_db=-np.inf
    )

    # B. Generate samples and estimate empirical correlation
    S = 20000  # Number of samples; larger S gives more stable empirical correlation
    H_iso, beta_iso, dist_iso = gen_iso.generate(S)
    H_iso_small = H_iso / np.sqrt(beta_iso)[:, None]   # Strip out large-scale fading
    R_iso_emp = empirical_R(H_iso_small, remove_mean=False)

    H_dir, beta_dir, dist_dir = gen_dir.generate(S)
    H_small = H_dir / np.sqrt(beta_dir)[:, None]       # Strip out large-scale fading
    R_dir_emp = empirical_R(H_small, remove_mean=True)

    # C. Theoretical / approximate correlation
    x = gen_iso.x
    k = gen_iso.k
    theta0 = gen_dir.aoa_mean
    sigma = gen_dir.aoa_spread
    R_iso_th = R_theory_isotropic(x, k)
    R_dir_th = R_theory_directional(x, theta0, sigma, k)

    # D. Plot R(Δx) (using port 0 as reference): theory vs. empirical comparison
    dx = x - x[0]
    def mae(a,b): return np.mean(np.abs(a-b))

    r_iso_th_line  = R_iso_th[0, :]
    r_iso_emp_line = R_iso_emp[0, :]
    r_dir_th_line  = R_dir_th[0, :]
    r_dir_emp_line = R_dir_emp[0, :]

    print(f"[Isotropic]   MAE Real={mae(np.real(r_iso_th_line), np.real(r_iso_emp_line)):.3e}, "
          f"MAE |.|={mae(np.abs(r_iso_th_line), np.abs(r_iso_emp_line)):.3e}")
    print(f"[Directional] MAE Real={mae(np.real(r_dir_th_line), np.real(r_dir_emp_line)):.3e}, "
          f"MAE |.|={mae(np.abs(r_dir_th_line), np.abs(r_dir_emp_line)):.3e}")
    
    # Visualize individual samples
    gen_iso.visualize(H_iso, beta_iso, dist_iso, sample_idx=1)
    gen_dir.visualize(H_dir, beta_dir, dist_dir, sample_idx=1)

    # Isotropic: theory vs. empirical correlation
    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    plt.plot(dx, np.real(r_iso_th_line), 'k-', lw=2, label='Theory Re{R}')
    plt.plot(dx, np.real(r_iso_emp_line), 'r--', lw=1.5, label='Empirical Re{R}')
    plt.title('Isotropic: Real part vs Δx'); plt.xlabel('Δx (λ)'); plt.ylabel('Re{R}')
    plt.grid(True); plt.legend()
    plt.subplot(1,2,2)
    plt.plot(dx, np.abs(r_iso_th_line), 'k-', lw=2, label='Theory |R|')
    plt.plot(dx, np.abs(r_iso_emp_line), 'b--', lw=1.5, label='Empirical |R|')
    plt.title('Isotropic: Magnitude vs Δx'); plt.xlabel('Δx (λ)'); plt.ylabel('|R|')
    plt.grid(True); plt.legend()
    plt.tight_layout(); plt.show()

    # Directional AoA: theory vs. empirical correlation
    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    plt.plot(dx, np.real(r_dir_th_line), 'k-', lw=2, label='Approx Re{R}')
    plt.plot(dx, np.real(r_dir_emp_line), 'r--', lw=1.5, label='Empirical Re{R}')
    plt.title(f'Directional: Real vs Δx (θ0={np.rad2deg(theta0):.1f}°, σ={np.rad2deg(sigma):.1f}°)')
    plt.xlabel('Δx (λ)'); plt.ylabel('Re{R}'); plt.grid(True); plt.legend()
    plt.subplot(1,2,2)
    plt.plot(dx, np.abs(r_dir_th_line), 'k-', lw=2, label='Approx |R|')
    plt.plot(dx, np.abs(r_dir_emp_line), 'b--', lw=1.5, label='Empirical |R|')
    plt.title(f'Directional: Magnitude vs Δx (θ0={np.rad2deg(theta0):.1f}°, σ={np.rad2deg(sigma):.1f}°)')
    plt.xlabel('Δx (λ)'); plt.ylabel('|R|'); plt.grid(True); plt.legend()
    plt.tight_layout(); plt.show()

    # E. Empirical correlation matrix heatmaps: isotropic vs. directional AoA
    vmax_real, vmin_real = 1.0, -1.0
    plt.figure(figsize=(10,8))
    plt.subplot(2,2,1)
    plt.imshow(np.real(R_iso_emp), cmap='Blues_r', vmin=vmin_real, vmax=vmax_real)
    plt.title('Isotropic: Re{R} (empirical)'); plt.xlabel('Port'); plt.ylabel('Port'); plt.colorbar(fraction=0.046)
    plt.subplot(2,2,2)
    plt.imshow(np.real(R_dir_emp), cmap='Blues_r', vmin=vmin_real, vmax=vmax_real)
    plt.title('Directional: Re{R} (empirical)'); plt.xlabel('Port'); plt.ylabel('Port'); plt.colorbar(fraction=0.046)
    plt.subplot(2,2,3)
    plt.imshow(np.abs(R_iso_emp), cmap='viridis', vmin=0, vmax=1)
    plt.title('Isotropic: |R| (empirical)'); plt.xlabel('Port'); plt.ylabel('Port'); plt.colorbar(fraction=0.046)
    plt.subplot(2,2,4)
    plt.imshow(np.abs(R_dir_emp), cmap='viridis', vmin=0, vmax=1)
    plt.title('Directional: |R| (empirical)'); plt.xlabel('Port'); plt.ylabel('Port'); plt.colorbar(fraction=0.046)
    plt.tight_layout(); plt.show()

    # F. Diagonal-average correlation coefficient r[m]: isotropic vs. directional AoA
    r_iso_emp_diag = empirical_r_diagonal_avg(H_iso_small, remove_mean=False)
    r_dir_emp_diag = empirical_r_diagonal_avg(H_small, remove_mean=True)
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(r_iso_emp_diag.real, 'b-o', markersize=4, label='Isotropic r[m]')
    plt.title('Isotropic: Diagonal Average Correlation r[m]'); plt.xlabel('m'); plt.ylabel('r[m]'); plt.grid(True); plt.legend()
    plt.subplot(1,2,2)
    plt.plot(r_dir_emp_diag.real, 'r-o', markersize=4, label='Directional r[m]')
    plt.title('Directional: Diagonal Average Correlation r[m]'); plt.xlabel('m'); plt.ylabel('r[m]'); plt.grid(True); plt.legend()
    plt.tight_layout(); plt.show()