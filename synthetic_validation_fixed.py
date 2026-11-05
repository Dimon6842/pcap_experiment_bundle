"""Corrected synthetic sanity-check for the spectral-gap pipeline.

This script is ONLY a software/method validation. Its numbers must not be
reported as evidence about real network traffic or in the final article.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.signal import periodogram, lombscargle
from sklearn.metrics import roc_auc_score

OUT = Path(__file__).resolve().parent
DT = 0.1
DURATION = 200.0
N = int(DURATION / DT)
T = (np.arange(N) + 0.5) * DT
FREQ_SEARCH = np.arange(0.5, 1.5001, 0.02)
FREQ_SCORE = (FREQ_SEARCH >= 0.8) & (FREQ_SEARCH <= 1.2)
METHODS = ("fft_linear_interpolation", "fft_zero_fill", "lomb_scargle")


def make_background(rng):
    """Poisson counts with a nonconstant, positive, slowly varying rate."""
    rho, sigma_eps = 0.82, 0.075
    ar = np.zeros(N, dtype=float)
    eps = rng.normal(0.0, sigma_eps, N)
    for i in range(1, N):
        ar[i] = rho * ar[i - 1] + eps[i]
    stationary_var = sigma_eps**2 / (1.0 - rho**2)
    seasonal = 1.0 + 0.18*np.sin(2*np.pi*T/17.0) + 0.10*np.sin(2*np.pi*T/4.7)
    rate = 20.0 * seasonal * np.exp(ar - 0.5*stationary_var)
    # Keep a physically meaningful positive count rate; critically, do not
    # apply the previous erroneous lower bound of 40 packets/s.
    rate = np.clip(rate, 0.1, None)
    return rng.poisson(rate * DT).astype(float), rate


def make_attack(rng):
    active = (T < 100.0) & ((T % 1.0) < 0.1)
    extra = np.where(active, 25.0, 0.0)
    return rng.poisson(extra * DT).astype(float)


def mask_for(condition, duration_s, rng):
    keep = np.ones(N, dtype=bool)
    if condition == "none" or duration_s <= 0:
        return keep
    length = max(1, int(round(duration_s / DT)))
    if condition == "active_block":
        start = int(round(((100.0-duration_s)/2.0)/DT))
        keep[start:start+length] = False
    elif condition == "quiet_block":
        start = int(round((100.0+(100.0-duration_s)/2.0)/DT))
        keep[start:start+length] = False
    elif condition == "boundary_block":
        start = int(round((100.0-duration_s/2.0)/DT))
        keep[start:start+length] = False
    elif condition == "random_points":
        keep[rng.choice(N, size=length, replace=False)] = False
    else:
        raise ValueError(condition)
    return keep


def spectral_score(x, keep, method):
    t_obs = T[keep]
    y_obs = np.asarray(x[keep], dtype=float)
    if len(y_obs) < 20:
        return float("nan"), float("nan")
    y_obs -= np.mean(y_obs)
    if method == "lomb_scargle":
        power = lombscargle(t_obs, y_obs, 2*np.pi*FREQ_SEARCH,
                            normalize=True, floating_mean=True)
        freq = FREQ_SEARCH
    else:
        y = np.zeros(N, dtype=float)
        if method == "fft_linear_interpolation":
            y = np.interp(T, t_obs, y_obs)
        elif method == "fft_zero_fill":
            y[keep] = y_obs
        else:
            raise ValueError(method)
        freq, power = periodogram(y, fs=1/DT, window="hann",
                                  detrend="constant", scaling="spectrum",
                                  nfft=4*N)
    band = (freq >= 0.8) & (freq <= 1.2)
    broad = (freq >= 0.5) & (freq <= 1.5)
    peak_idx = np.flatnonzero(broad)[np.argmax(power[broad])]
    peak_score = float(np.max(power[band]) / (np.median(power[broad]) + 1e-12))
    return peak_score, float(freq[peak_idx])


def run(n_pairs=40, seed=20261030):
    rng = np.random.default_rng(seed)
    backgrounds, attacks, rate_means, rate_mins, rate_maxs = [], [], [], [], []
    for _ in range(n_pairs):
        bg, rate = make_background(rng)
        backgrounds.append(bg)
        attacks.append(make_attack(rng))
        rate_means.append(float(np.mean(rate)))
        rate_mins.append(float(np.min(rate)))
        rate_maxs.append(float(np.max(rate)))

    scenarios = [("none", 0.0)]
    for place in ("active_block", "quiet_block"):
        scenarios.append((place, 20.0))

    rows = []
    for condition, duration in scenarios:
        for method in METHODS:
            normal, abnormal, errors = [], [], []
            for i in range(n_pairs):
                mask_rng = np.random.default_rng(seed + 10007*i + int(duration*100) + sum(map(ord, condition)))
                keep = mask_for(condition, duration, mask_rng)
                s0, _ = spectral_score(backgrounds[i], keep, method)
                s1, f1 = spectral_score(backgrounds[i] + attacks[i], keep, method)
                normal.append(s0); abnormal.append(s1)
                if np.isfinite(f1) and f1 > 0:
                    errors.append(abs(1.0/f1 - 1.0) * 100.0)
            normal = np.asarray(normal); abnormal = np.asarray(abnormal)
            split = n_pairs // 2
            y_test = np.r_[np.zeros(n_pairs-split, dtype=int), np.ones(n_pairs-split, dtype=int)]
            scores_test = np.r_[normal[split:], abnormal[split:]]
            auc = float(roc_auc_score(y_test, scores_test))
            threshold = float(np.quantile(normal[:split], .95, method="higher"))
            rows.append({
                "scenario": condition, "gap_duration_s": duration, "missing_fraction": duration/DURATION,
                "method": method, "test_auc": auc,
                "test_tpr": float(np.mean(abnormal[split:] > threshold)),
                "test_fpr": float(np.mean(normal[split:] > threshold)),
                "median_period_error_pct": float(np.nanmedian(errors)) if errors else np.nan,
                "n_calibration_normal": split, "n_test_normal": n_pairs-split,
                "n_test_attack": n_pairs-split,
                "data_type": "synthetic_sanity_check_only_not_real_PCAP",
            })
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "synthetic_validation_fixed.csv", index=False)
    meta = {
        "seed": seed, "pairs": n_pairs, "dt_s": DT, "duration_s": DURATION,
        "background_rate_mean_over_pairs_pps": float(np.mean(rate_means)),
        "background_rate_min_pps": float(np.min(rate_mins)),
        "background_rate_max_pps": float(np.max(rate_maxs)),
        "background_count_mean_per_bin": float(np.mean([np.mean(x) for x in backgrounds])),
        "bug_fix": "removed erroneous np.clip(rate, 40, None), which flattened every synthetic rate to 40 pps",
        "limitation": "software sanity check only; no claims about measured network traffic",
    }
    (OUT / "synthetic_validation_fixed_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))
    print(df[df.method == "lomb_scargle"].round(4).to_string(index=False))
    return df

if __name__ == "__main__":
    run()
