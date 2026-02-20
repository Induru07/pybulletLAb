from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import math


@dataclass
class PFConfig:
    n: int = 200

    # Bruits "motion-dependent" (coefficients)
    trans_noise_per_m: float = 0.02   # m de bruit par m parcouru
    rot_noise_per_rad: float = 0.02   # rad de bruit par rad tourné

    # Planchers (quasi zéro à l'arrêt)
    trans_noise_min: float = 0.001
    rot_noise_min: float = 0.001

    meas_std: float = 0.25



class ParticleFilter:
    """
    Particules: [x, y, theta], poids w
    """
    def __init__(self, cfg: PFConfig):
        self.cfg = cfg
        self.p = np.zeros((cfg.n, 3), dtype=np.float32)
        self.w = np.ones(cfg.n, dtype=np.float32) / cfg.n

    def estimate_map(self) -> tuple[float, float, float]:
        i = int(np.argmax(self.w))
        return float(self.p[i, 0]), float(self.p[i, 1]), float(self.p[i, 2])


    def init_uniform(self, x_min, x_max, y_min, y_max):
        self.p[:, 0] = np.random.uniform(x_min, x_max, size=self.cfg.n)
        self.p[:, 1] = np.random.uniform(y_min, y_max, size=self.cfg.n)
        self.p[:, 2] = np.random.uniform(-math.pi, math.pi, size=self.cfg.n)
        self.w.fill(1.0 / self.cfg.n)

    def init_gaussian(self, x: float, y: float, theta: float, std_xy: float = 0.5, std_th: float = 0.5):
        self.p[:, 0] = np.random.normal(x, std_xy, size=self.cfg.n)
        self.p[:, 1] = np.random.normal(y, std_xy, size=self.cfg.n)
        self.p[:, 2] = np.random.normal(theta, std_th, size=self.cfg.n)
        self.p[:, 2] = (self.p[:, 2] + math.pi) % (2 * math.pi) - math.pi
        self.w.fill(1.0 / self.cfg.n)

    def predict(self, dC: float, dT: float):
        # bruit du mouvement (if dC=dT=0 => bruit ~ min)
        sig_t = max(self.cfg.trans_noise_min, self.cfg.trans_noise_per_m * abs(dC))
        sig_r = max(self.cfg.rot_noise_min, self.cfg.rot_noise_per_rad * abs(dT))

        dC_noisy = dC + np.random.normal(0.0, sig_t, size=self.cfg.n)
        dT_noisy = dT + np.random.normal(0.0, sig_r, size=self.cfg.n)

        self.p[:, 2] += dT_noisy
        self.p[:, 2] = (self.p[:, 2] + math.pi) % (2 * math.pi) - math.pi

        self.p[:, 0] += dC_noisy * np.cos(self.p[:, 2])
        self.p[:, 1] += dC_noisy * np.sin(self.p[:, 2])


    def update(self, z: np.ndarray, z_hat_particles: np.ndarray):
        """
        z: (K,) mesures
        z_hat_particles: (N,K) scans simulés pour chaque particule
        """
        # Gaussian likelihood
        diff = z_hat_particles - z[None, :]
        sigma2 = self.cfg.meas_std ** 2
        ll = -0.5 * np.sum((diff * diff) / sigma2, axis=1)  # log-likelihood
        ll -= np.max(ll)  # stabilité
        w = np.exp(ll).astype(np.float32)
        w += 1e-12
        self.w = w / np.sum(w)

    def neff(self) -> float:
        return 1.0 / float(np.sum(self.w * self.w))

    def resample(self):
        import numpy as np

        w = self.w.astype(np.float64)

        # Normalisation robuste
        s = np.sum(w)
        if s <= 0:
            w = np.ones_like(w) / len(w)
        else:
            w = w / s

        cumsum = np.cumsum(w)
        cumsum[-1] = 1.0  # sécurité numérique

        n = len(w)

        # Systematic resampling (positions dans [0,1) -> jamais 1.0)
        u0 = np.random.random() / n
        positions = u0 + np.arange(n) / n

        idx = np.searchsorted(cumsum, positions, side="right")
        idx = np.clip(idx, 0, n - 1)

        self.p = self.p[idx].copy()
        self.w[:] = 1.0 / n

    # ── Kidnapped-robot recovery ──
    def inject_random_particles(self, free_cells_xy: np.ndarray, frac: float = 0.05):
        """
        Replace *frac* of the lowest-weight particles with random samples
        drawn uniformly from *free_cells_xy* (Nx2 array of (x,y) positions).
        This gives the filter a chance to recover from kidnapping / severe drift
        without needing ground-truth.
        """
        n_inject = max(1, int(self.cfg.n * frac))
        if len(free_cells_xy) == 0:
            return
        # Pick the worst particles (lowest weight)
        worst_idx = np.argsort(self.w)[:n_inject]
        # Random free-space positions
        chosen = free_cells_xy[np.random.randint(0, len(free_cells_xy), size=n_inject)]
        self.p[worst_idx, 0] = chosen[:, 0]
        self.p[worst_idx, 1] = chosen[:, 1]
        self.p[worst_idx, 2] = np.random.uniform(-math.pi, math.pi, size=n_inject)
        self.w[worst_idx] = 1.0 / self.cfg.n  # equal prior weight


    def estimate(self) -> tuple[float, float, float]:
        x = float(np.sum(self.p[:, 0] * self.w))
        y = float(np.sum(self.p[:, 1] * self.w))
        # moyenne circulaire
        s = float(np.sum(np.sin(self.p[:, 2]) * self.w))
        c = float(np.sum(np.cos(self.p[:, 2]) * self.w))
        theta = math.atan2(s, c)
        return x, y, theta
