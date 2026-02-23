from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Sequence, Optional

import numpy as np
import matplotlib.pyplot as plt

def _interp_nans(a):
    a = np.asarray(a, float).copy()
    nan_mask = np.isnan(a)
    if nan_mask.any():
        a[nan_mask] = np.interp(
            np.flatnonzero(nan_mask),
            np.flatnonzero(~nan_mask),
            a[~nan_mask],
        )
    return a

def _wrap_angle(x):
    return (x + np.pi) % (2 * np.pi) - np.pi

def _even_minute_mask(n_steps: int, dt: float) -> np.ndarray:
    steps_per_min = int(round(60 / dt))
    return (np.arange(n_steps) // steps_per_min) % 2 == 0

def _odd_minute_mask(n_steps: int, dt: float) -> np.ndarray:
    return ~_even_minute_mask(n_steps, dt)

def angdiff(a, b):
    diff = (a - b) % (2.0 * np.pi)
    return np.minimum(diff, 2.0 * np.pi - diff)

@dataclass
class EBCSimulator:
    n_neurons: int = 300          #spike matric-- should this be binarized matrix? or is it enough to have spiketimes in seconds for each cell?
    arena_size: float = 1.43          # metres (square)   ## 1.43m x 1.43m in practice
    dt: float = 0.01                 # seconds     #what is this? a binsize or?
    duration: float = 600.0          # seconds (10 min)      ###49 mins

    # tuning & firing‑rate parameters
    baseline_rate: float = 0.1       # Hz
    sigma_angle: float = np.deg2rad(20)
    sigma_distance: float = 0.1      # metres
    max_rate_range: Tuple[float, float] = (15.0, 25.0)
    pref_distance_range: Tuple[float, float] = (0.0, 0.5)

    # movement parameters
    speed_mean: float = 0.25         # m/s
    speed_std: float = 0.05
    angvel_std: float = np.deg2rad(60)  # rad/s√s

    seed: Optional[int] = None

    rng: np.random.Generator = field(init=False, repr=False)

    # -----------------------------------------------------------------
    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)
        self._init_neurons()

    # -----------------------------------------------------------------
    def run(self) -> Tuple[List[np.ndarray], Dict[str, np.ndarray]]:
        """Run simulation and return (spikes, state)."""
        n_steps = int(self.duration / self.dt)

        pos = np.zeros((n_steps, 2))          # (x, y) -- found in animal_location in column, 0 and 1
        hd = np.zeros(n_steps)                # head direction (in degree but needs to converted rad) -- found in 'D Allo_head_direction'
        d_wall = np.zeros(n_steps)            # distance to nearest wall (cm) - found in 'V Distance to wall'
        phi_rel = np.zeros(n_steps)           # egocentric bearing (in degree but needs to converted to rad)-- found in 'U Angle_to_wall'

        # initial condition
        pos[0] = self.rng.uniform(0.2, self.arena_size - 0.2, 2)
        hd[0] = self.rng.uniform(-np.pi, np.pi)

        speed = self.rng.normal(self.speed_mean, self.speed_std, n_steps)
        angvel = self.rng.normal(0.0, self.angvel_std * math.sqrt(self.dt), n_steps)

        for t in range(1, n_steps):
            hd[t] = _wrap_angle(hd[t - 1] + angvel[t])
            step = speed[t] * self.dt
            dx, dy = step * math.cos(hd[t - 1]), step * math.sin(hd[t - 1])
            new_pos = pos[t - 1] + (dx, dy)

            # reflections
            if new_pos[0] < 0:
                new_pos[0] *= -1
                hd[t] = _wrap_angle(np.pi - hd[t])
            if new_pos[0] > self.arena_size:
                new_pos[0] = 2 * self.arena_size - new_pos[0]
                hd[t] = _wrap_angle(np.pi - hd[t])
            if new_pos[1] < 0:
                new_pos[1] *= -1
                hd[t] = _wrap_angle(-hd[t])
            if new_pos[1] > self.arena_size:
                new_pos[1] = 2 * self.arena_size - new_pos[1]
                hd[t] = _wrap_angle(-hd[t])

            pos[t] = new_pos
            d_wall[t], phi_rel[t] = self._closest_wall_egocentric(new_pos, hd[t])

        # instantaneous firing‑rates (n_neurons × n_steps)
        dphi = _wrap_angle(phi_rel[None, :] - self.phi_pref[:, None])
        dd = d_wall[None, :] - self.d_pref[:, None]
        rate = (
            self.baseline_rate
            + self.max_rate[:, None]
            * np.exp(-0.5 * (dphi / self.sigma_angle) ** 2)
            * np.exp(-0.5 * (dd / self.sigma_distance) ** 2)
        )

        prob = rate * self.dt
        rand = self.rng.random(rate.shape)
        spikes = [np.nonzero(rand[i] < prob[i])[0] * self.dt for i in range(self.n_neurons)]

        state = {"pos": pos, "head_dir": hd, "d_wall": d_wall, "phi_rel": phi_rel}
        return spikes, state

    def _init_neurons(self):
        self.phi_pref = self.rng.uniform(-np.pi, np.pi, self.n_neurons)
        self.d_pref = self.rng.uniform(*self.pref_distance_range, self.n_neurons)
        self.max_rate = self.rng.uniform(*self.max_rate_range, self.n_neurons)

    def _closest_wall_egocentric(self, pos, hd):
        x, y = pos
        L = self.arena_size
        vectors = np.array([[-x, 0], [L - x, 0], [0, -y], [0, L - y]])
        dists = np.linalg.norm(vectors, axis=1)
        idx = dists.argmin()
        vec = vectors[idx]
        return dists[idx], _wrap_angle(math.atan2(vec[1], vec[0]) - hd)


def compute_rate_maps(
    spikes: Sequence[np.ndarray],
    state: Dict[str, np.ndarray],
    dt: float,
    *,
    distance_bins: np.ndarray | None = None,
    angle_bins: np.ndarray | None = None,
    even_minutes_only: bool = False,
    time_mask: Optional[np.ndarray] = None,
    debug: bool = False,
):
    """Estimate 2D Poisson rate maps over (distance, bearing).

    Notes
    -----
    - Distances and bins must be in the SAME unit (recommend: metres).
    - Any NaNs in state are automatically excluded from the mask.
    """
    if "d_wall" in state and "phi_rel" in state:
        d = np.asarray(state["d_wall"], float)
        phi = np.asarray(state["phi_rel"], float)
    elif "d_bait" in state and "phi_bait" in state:
        d = np.asarray(state["d_bait"], float)
        phi = np.asarray(state["phi_bait"], float)
    else:
        raise KeyError("State dict must contain either (d_wall, phi_rel) or (d_bait, phi_bait)")

    n_steps = len(d)

    # build mask
    if time_mask is not None:
        time_mask = np.asarray(time_mask, bool)
    elif even_minutes_only:
        time_mask = _even_minute_mask(n_steps, dt)
    else:
        time_mask = np.ones(n_steps, bool)

    # ALWAYS remove invalid state bins (important for chasing marker dropouts)
    valid_state = np.isfinite(d) & np.isfinite(phi)
    time_mask = time_mask & valid_state

    # set up bins if not provided
    if distance_bins is None:
        distance_bins = np.linspace(0.0, 0.6, 25)
    if angle_bins is None:
        angle_bins = np.linspace(-np.pi, np.pi, 61)

    # occupancy
    occ_counts, _, _ = np.histogram2d(d[time_mask], phi[time_mask], bins=[distance_bins, angle_bins])
    occ = occ_counts * dt  # seconds per bin

    if debug:
        total_occ = occ.sum()
        expected = time_mask.sum() * dt
        print(f"[DBG] Mask → {time_mask.sum()} / {n_steps} bins ({time_mask.mean()*100:.1f}%)")
        print(f"[DBG] Occ sum = {total_occ:.3f}s vs expected {expected:.3f}s")

    # map spikes to masked indices
    mask_idx = np.nonzero(time_mask)[0]
    idx_lookup = -np.ones(n_steps, int)
    idx_lookup[mask_idx] = np.arange(mask_idx.size)
    d_masked, phi_masked = d[time_mask], phi[time_mask]

    n_neurons = len(spikes)
    n_d = len(distance_bins) - 1
    n_phi = len(angle_bins) - 1
    rate_maps = np.zeros((n_neurons, n_d, n_phi), float)

    for n, spk in enumerate(spikes):
        spk = np.asarray(spk, float)
        idx = np.floor(spk / dt).astype(int)
        idx = idx[(idx >= 0) & (idx < n_steps)]
        idx = idx_lookup[idx]
        idx = idx[idx >= 0]

        d_spk = d_masked[idx]
        phi_spk = phi_masked[idx]
        spk_counts, _, _ = np.histogram2d(d_spk, phi_spk, bins=[distance_bins, angle_bins])

        with np.errstate(divide="ignore", invalid="ignore"):
            rate_maps[n] = np.where(occ > 0, spk_counts / occ, 0.0)

        if debug and n < 5:
            print(f"[DBG] Neuron {n}: spikes_in_mask={int(spk_counts.sum())}")

    return rate_maps, distance_bins, angle_bins


def decode_ebc(
    spikes: Sequence[np.ndarray],
    state: Dict[str, np.ndarray],
    dt: float,
    rate_maps: np.ndarray,
    distance_bins: np.ndarray,
    angle_bins: np.ndarray,
    *,
    odd_minutes_only: bool = False,
    time_mask: Optional[np.ndarray] = None,
    window: float = 0.1,
    stride: Optional[float] = None,
    mask_full_window: bool = True,
    method: str = "map",
    debug: bool = False,
):
    """Decode (distance, bearing) with a Poisson Naive Bayes model.

    Key improvements for chasing:
    - Drops NaN state bins automatically.
    - Can require the *entire* decoding window to be valid (mask_full_window=True).
    - Allows a smaller stride than the window (overlapping windows) via `stride`.
    - Optionally returns posterior mean instead of hard MAP (method='mean').
    """
    n_neurons, n_d, n_phi = rate_maps.shape

    if "d_wall" in state and "phi_rel" in state:
        d = np.asarray(state["d_wall"], float)
        phi = np.asarray(state["phi_rel"], float)
    elif "d_bait" in state and "phi_bait" in state:
        d = np.asarray(state["d_bait"], float)
        phi = np.asarray(state["phi_bait"], float)
    else:
        raise KeyError("State dict must contain either (d_wall, phi_rel) or (d_bait, phi_bait)")

    n_steps = len(d)

    steps_per_window = max(1, int(round(window / dt)))
    stride_steps = max(1, int(round((stride if stride is not None else window) / dt)))

    # base time mask
    if time_mask is not None:
        base_mask = np.asarray(time_mask, bool).copy()
    elif odd_minutes_only:
        base_mask = _odd_minute_mask(n_steps, dt)
    else:
        base_mask = np.ones(n_steps, bool)

    # ALWAYS remove invalid state bins
    base_mask &= np.isfinite(d) & np.isfinite(phi)

    # choose valid window starts
    if mask_full_window:
        # start is valid only if ALL bins inside window are valid
        # (convolution-based mask erosion)
        kernel = np.ones(steps_per_window, dtype=np.int32)
        ok = np.convolve(base_mask.astype(np.int32), kernel, mode="valid") == steps_per_window
        candidate_starts = np.flatnonzero(ok)
        starts = candidate_starts[::stride_steps]
    else:
        starts = np.arange(0, n_steps - steps_per_window + 1, stride_steps)
        starts = starts[base_mask[starts]]

    if debug:
        print(f"[DBG] decode_ebc: window={window}s ({steps_per_window} bins), stride={stride_steps} bins")
        print(f"[DBG] valid starts: {len(starts)}")

    t_centres = starts * dt + (steps_per_window * dt) / 2.0

    lam = rate_maps.reshape(n_neurons, n_d * n_phi)
    lam_dt = lam * window  # expected spikes in window

    decoded_d = np.empty(starts.size)
    decoded_phi = np.empty(starts.size)

    d_centres = (distance_bins[:-1] + distance_bins[1:]) / 2.0
    phi_centres = _wrap_angle((angle_bins[:-1] + angle_bins[1:]) / 2.0)

    use_mean = method.lower() in {"mean", "posterior_mean", "expected"}

    for i, s in enumerate(starts):
        e = s + steps_per_window
        k = np.array([((sp >= s * dt) & (sp < e * dt)).sum() for sp in spikes], dtype=float)

        # log P(k | state) up to constant: sum_i [k_i log λ_i - λ_i]
        log_likelihood = k[:, None] * np.log(lam_dt + 1e-12) - lam_dt
        log_l_surface = log_likelihood.sum(axis=0)  # (n_d*n_phi,)

        if not use_mean:
            best = int(log_l_surface.argmax())
            id_d, id_phi = divmod(best, n_phi)
            decoded_d[i] = d_centres[id_d]
            decoded_phi[i] = phi_centres[id_phi]
        else:
            # stable softmax
            m = float(np.max(log_l_surface))
            p = np.exp(log_l_surface - m)
            p_sum = float(p.sum())
            if p_sum <= 0 or not np.isfinite(p_sum):
                best = int(log_l_surface.argmax())
                id_d, id_phi = divmod(best, n_phi)
                decoded_d[i] = d_centres[id_d]
                decoded_phi[i] = phi_centres[id_phi]
            else:
                p /= p_sum
                p2 = p.reshape(n_d, n_phi)
                decoded_d[i] = float((p2 * d_centres[:, None]).sum())
                sin_mean = float((p2 * np.sin(phi_centres)[None, :]).sum())
                cos_mean = float((p2 * np.cos(phi_centres)[None, :]).sum())
                decoded_phi[i] = float(_wrap_angle(np.arctan2(sin_mean, cos_mean)))

    return {"t": t_centres, "decoded_d": decoded_d, "decoded_phi": decoded_phi}

# ---------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------

def plot_ebc(
    spikes: Sequence[np.ndarray],
    state: Dict[str, np.ndarray],
    dt: float,
    *,
    neuron_indices: int | Sequence[int] = 0,
    distance_bins: np.ndarray | None = None,
    angle_bins: np.ndarray | None = None,
    figsize: Tuple[int, int] = (14, 4),
):
    """Polar occupancy / spike / rate maps for one or more neurons."""
    if distance_bins is None:
        distance_bins = np.linspace(0.0, 0.6, 25)
    if angle_bins is None:
        angle_bins = np.linspace(-np.pi, np.pi, 61)
    if isinstance(neuron_indices, int):
        neuron_indices = [neuron_indices]

    d = state["d_wall"]
    phi = state["phi_rel"]

    occ_counts, _, _ = np.histogram2d(d, phi, bins=[distance_bins, angle_bins])
    occ = occ_counts * dt

    # prep extended theta axis (0–2π + seam)
    r_centres = (distance_bins[:-1] + distance_bins[1:]) / 2
    theta_centres = (angle_bins[:-1] + angle_bins[1:]) / 2 + np.pi / 2
    theta_ext = np.hstack([theta_centres, theta_centres[0] + 2 * np.pi])

    for ni in neuron_indices:
        idx = np.clip((spikes[ni] / dt).astype(int), 0, len(d) - 1)
        d_spk = d[idx]
        phi_spk = phi[idx]
        spk_counts, _, _ = np.histogram2d(d_spk, phi_spk, bins=[distance_bins, angle_bins])
        spk_ext = np.hstack([spk_counts, spk_counts[:, :1]])
        occ_ext = np.hstack([occ, occ[:, :1]])
        with np.errstate(divide="ignore", invalid="ignore"):
            rate_ext = np.where(occ_ext > 0, spk_ext / occ_ext, np.nan)

        data_stack = [occ_ext, spk_ext, rate_ext]
        titles = ["Occupancy (s)", "Spike count", "Rate (Hz)"]

        fig = plt.figure(figsize=figsize)
        fig.suptitle(f"Neuron {ni} egocentric boundary tuning", fontsize=14)

        theta_grid, r_grid = np.meshgrid(theta_ext, r_centres)
        for k, (dat, ttl) in enumerate(zip(data_stack, titles), 1):
            ax = fig.add_subplot(1, 3, k, projection="polar")
            pcm = ax.pcolormesh(theta_grid, r_grid, dat, shading="auto")
            ax.set_theta_zero_location("N")
            ax.set_theta_direction(-1)
            ax.set_ylim(r_centres[0], r_centres[-1])
            ax.set_title(ttl, pad=12)
            fig.colorbar(pcm, ax=ax, shrink=0.78, pad=0.05)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

def save_decode_summary(filename: str,
                        decode: Dict[str, np.ndarray],
                        state: Dict[str, np.ndarray],
                        dt: float) -> None:
    """
    Save the decoding results plus ground-truth head & wall locations.

    Parameters
    ----------
    filename : str
        Name of the .npz file to write.
    decode : dict
        Output of `decode_ebc` (must contain 't', 'decoded_d', 'decoded_phi').
    state : dict
        The `state` dict returned by `EBCSimulator.run()`.
    dt : float
        Simulation time-step (seconds).
    """
    # indices of the time bins that were decoded
    t = decode["t"]
    idx = np.clip((t / dt).astype(int), 0, len(state["pos"]) - 1)

    # ground-truth head position & direction
    head_pos = state["pos"][idx]            # shape (N, 2)
    head_dir = state["head_dir"][idx]       # shape (N,)

    # true egocentric wall distance/bearing at those bins
    d_wall = state["d_wall"][idx]           # (N,)
    phi_rel = state["phi_rel"][idx]         # (N,)

    # convert them to world-coordinates of the nearest wall point
    world_bearing = head_dir + phi_rel      # absolute bearing
    wall_xy = head_pos + d_wall[:, None] * np.stack(
        [np.cos(world_bearing), np.sin(world_bearing)], axis=1
    )                                       # shape (N, 2)

    # write compressed .npz
    np.savez_compressed(
        filename,
        t=t,
        decoded_d=decode["decoded_d"],
        decoded_phi=decode["decoded_phi"],
        head_pos=head_pos,
        head_dir=head_dir,
        wall_xy=wall_xy,
    )
    print(f"Saved decode summary → {filename!s}")

# ---------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------

def run_encode_decode_demo(seed: int | None = 0):
    """Run 10‑min simulation, encode from even minutes, decode odd minutes."""
    sim = EBCSimulator(seed=seed)
    spikes, state = sim.run()

    # encode
    rmaps, d_bins, th_bins = compute_rate_maps(
        spikes, state, sim.dt, even_minutes_only=True)

    # decode
    decode = decode_ebc(
        spikes, state, sim.dt, rmaps, d_bins, th_bins,
        odd_minutes_only=True)

    # compare to ground truth for decoded bins
    idx_true = np.floor(decode["t"] / sim.dt).astype(int)
    err_d = np.abs(state["d_wall"][idx_true] - decode["decoded_d"])
    err_phi = np.abs(_wrap_angle(state["phi_rel"][idx_true] - decode["decoded_phi"]))

    print("\n=== 10‑minute encode→decode demo ===")
    print(f"Mean |error| distance: {err_d.mean():.3f} m")
    print(f"Mean |error| bearing : {np.degrees(err_phi).mean():.1f} deg")
    print("(distance ∈ [0, 0.5] m; bearing 0 ≈ straight ahead)")

    # quick plot of first neuron rate map for sanity
    plot_ebc(spikes, state, sim.dt, neuron_indices=0,
             distance_bins=d_bins, angle_bins=th_bins)

    save_decode_summary("decode_summary.npz", decode, state, sim.dt)


if __name__ == "__main__":
    run_encode_decode_demo()
