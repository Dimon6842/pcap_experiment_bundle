#!/usr/bin/env python3
"""Main PCAP experiment for the article on periodic network anomalies and gaps.

Input archives (official UTSA-2021 release):
  PCAPs_1.zip, PCAPs_2.zip, PCAPs_3.zip, Benign_Only.zip

The script deliberately keeps the analysis reproducible and auditable:
- classic PCAP and PCAPNG readers (Ethernet, VLAN, Linux cooked, raw IPv4/IPv6);
- TCP SYN packets without ACK are counted in fixed-width time bins;
- attack-phase alignment is estimated from the documented 200-s template
  (100 s with 0.1-s pulses each second, then 100 s quiet);
- complete 200-s windows are extracted;
- equal-duration gaps are placed in active, quiet, or boundary phases, and
  fragmented/random masks remove the same number of observations;
- FFT after interpolation, FFT after zero filling, and Lomb-Scargle are compared;
- a threshold is calibrated on the training part of benign captures only.

Important: no results should be interpreted until the dataset inventory and
phase-alignment diagnostics have been inspected. The script does not invent
missing labels; ambiguous filenames are reported as unlabelled.

Dependencies: numpy, pandas, scipy, scikit-learn; numba is optional and accelerates Lomb–Scargle.
"""
from __future__ import annotations
import argparse
import io
import json
import math
import re
import struct
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
import pandas as pd
from scipy.signal import periodogram, lombscargle
try:
    from numba import njit
except ImportError:  # use SciPy fallback if Numba is unavailable
    njit = None
from sklearn.metrics import roc_auc_score

WINDOW_S = 200.0
PERIOD_S = 1.0
ACTIVE_S = 100.0
PULSE_S = 0.1
GAP_DURATIONS = (5.0, 10.0, 20.0, 40.0)
METHODS = ("fft_linear", "fft_zero", "lomb_scargle")
F_SEARCH_MIN, F_SEARCH_MAX = 0.5, 1.5
F_SCORE_MIN, F_SCORE_MAX = 0.8, 1.2
SEED = 20261030


@dataclass
class Capture:
    archive: str
    member: str
    label: Optional[int]
    attack_rate: Optional[str]
    first_ts: float
    last_ts: float
    packet_count: int
    syn_times: np.ndarray
    linktypes: str

    @property
    def duration(self) -> float:
        return max(0.0, self.last_ts - self.first_ts)

    @property
    def capture_id(self) -> str:
        return f"{Path(self.archive).name}::{self.member}"


def infer_label(path: str) -> tuple[Optional[int], Optional[str]]:
    name = path.lower()
    if "benign" in name or "normal" in name:
        return 0, "benign"
    m = re.search(r"syn\s*[-_ ]?(\d+)", name, re.I)
    if m:
        return 1, f"Syn{m.group(1)}"
    if re.search(r"syn", name, re.I):
        return 1, "SYN_attack_rate_unspecified"
    # Slow-read/slow-get samples are not part of this SYN-pulse experiment.
    if "slowread" in name or "slow_read" in name or "slowget" in name:
        return None, "excluded_non_SYN_attack"
    return None, None


def _packet_is_syn_no_ack(pkt: memoryview, linktype: int) -> bool:
    """Check a captured frame for a TCP SYN=1, ACK=0 packet."""
    n = len(pkt)
    off = 0
    ethertype = None
    if linktype == 1:  # Ethernet
        if n < 14:
            return False
        ethertype = struct.unpack_from("!H", pkt, 12)[0]
        off = 14
        while ethertype in (0x8100, 0x88A8, 0x9100):
            if n < off + 4:
                return False
            ethertype = struct.unpack_from("!H", pkt, off + 2)[0]
            off += 4
    elif linktype == 113:  # Linux cooked capture v1
        if n < 16:
            return False
        ethertype = struct.unpack_from("!H", pkt, 14)[0]
        off = 16
    elif linktype == 276:  # Linux cooked capture v2
        if n < 20:
            return False
        ethertype = struct.unpack_from("!H", pkt, 0)[0]
        off = 20
    elif linktype in (101, 228, 229):  # raw IP / IPv4 / IPv6
        if n < 1:
            return False
        version = pkt[0] >> 4
        ethertype = 0x0800 if version == 4 else (0x86DD if version == 6 else 0)
        off = 0
    else:
        return False

    # IPv4
    if ethertype == 0x0800:
        if n < off + 20 or (pkt[off] >> 4) != 4:
            return False
        ihl = (pkt[off] & 0x0F) * 4
        if ihl < 20 or n < off + ihl or pkt[off + 9] != 6:
            return False
        frag = struct.unpack_from("!H", pkt, off + 6)[0]
        if frag & 0x1FFF:  # non-initial fragment has no TCP header
            return False
        tcp = off + ihl
    # IPv6 (basic extension-header traversal)
    elif ethertype == 0x86DD:
        if n < off + 40 or (pkt[off] >> 4) != 6:
            return False
        nxt = pkt[off + 6]
        tcp = off + 40
        for _ in range(8):
            if nxt == 6:
                break
            if nxt in (0, 43, 60, 135):
                if n < tcp + 2:
                    return False
                new_nxt = pkt[tcp]
                ext_len = (pkt[tcp + 1] + 1) * 8
                nxt = new_nxt
                tcp += ext_len
            elif nxt == 44:
                if n < tcp + 8:
                    return False
                new_nxt = pkt[tcp]
                frag_off = struct.unpack_from("!H", pkt, tcp + 2)[0]
                if frag_off & 0xFFF8:
                    return False
                nxt = new_nxt
                tcp += 8
            elif nxt == 51:
                if n < tcp + 2:
                    return False
                new_nxt = pkt[tcp]
                ext_len = (pkt[tcp + 1] + 2) * 4
                nxt = new_nxt
                tcp += ext_len
            else:
                return False
        if nxt != 6:
            return False
    else:
        return False

    if n < tcp + 20:
        return False
    data_offset = (pkt[tcp + 12] >> 4) * 4
    if data_offset < 20 or n < tcp + data_offset:
        return False
    flags = pkt[tcp + 13]
    return bool(flags & 0x02) and not bool(flags & 0x10)


def _parse_classic_pcap(blob: bytes) -> tuple[list[float], list[int], float, float, int]:
    if len(blob) < 24:
        raise ValueError("PCAP header is truncated")
    magic = blob[:4]
    magics = {
        b"\xd4\xc3\xb2\xa1": ("<", 1e-6),
        b"\xa1\xb2\xc3\xd4": (">", 1e-6),
        b"\x4d\x3c\xb2\xa1": ("<", 1e-9),
        b"\xa1\xb2\x3c\x4d": (">", 1e-9),
    }
    if magic not in magics:
        raise ValueError("Unknown classic PCAP magic")
    endian, ts_scale = magics[magic]
    _, _, _, _, _, _snaplen, linktype = struct.unpack_from(endian + "IHHIIII", blob, 0)
    pos = 24
    syn_times: list[float] = []
    linktypes: set[int] = {linktype}
    first_ts = math.inf
    last_ts = -math.inf
    packets = 0
    mv = memoryview(blob)
    while pos + 16 <= len(blob):
        sec, frac, incl_len, _orig_len = struct.unpack_from(endian + "IIII", mv, pos)
        pos += 16
        if incl_len <= 0 or pos + incl_len > len(blob):
            break
        ts = sec + frac * ts_scale
        first_ts = min(first_ts, ts)
        last_ts = max(last_ts, ts)
        pkt = mv[pos:pos+incl_len]
        if _packet_is_syn_no_ack(pkt, linktype):
            syn_times.append(ts)
        packets += 1
        pos += incl_len
    if not packets:
        raise ValueError("No readable packets in classic PCAP")
    return syn_times, sorted(linktypes), first_ts, last_ts, packets


def _parse_pcapng(blob: bytes) -> tuple[list[float], list[int], float, float, int]:
    pos = 0
    endian = None
    interfaces: list[tuple[int, float]] = []  # (linktype, seconds per tick)
    syn_times: list[float] = []
    first_ts, last_ts, packets = math.inf, -math.inf, 0
    linktypes: set[int] = set()
    while pos + 12 <= len(blob):
        block_type_raw = blob[pos:pos+4]
        if block_type_raw == b"\x0a\x0d\x0d\x0a":
            # Interface IDs are scoped to a PCAPNG section.
            interfaces = []
            if pos + 12 > len(blob):
                break
            bom = blob[pos+8:pos+12]
            if bom == b"\x4d\x3c\x2b\x1a":
                endian = "<"
            elif bom == b"\x1a\x2b\x3c\x4d":
                endian = ">"
            else:
                raise ValueError("Unknown PCAPNG byte-order magic")
            block_type = 0x0A0D0D0A
        else:
            if endian is None:
                raise ValueError("PCAPNG section header missing")
            block_type = struct.unpack_from(endian + "I", blob, pos)[0]
        block_len = struct.unpack_from(endian + "I", blob, pos+4)[0]
        if block_len < 12 or pos + block_len > len(blob):
            break
        body_start, body_end = pos+8, pos+block_len-4
        if block_type == 1:  # Interface Description Block
            if body_start + 8 <= body_end:
                linktype = struct.unpack_from(endian + "H", blob, body_start)[0]
                resolution = 1e-6
                opt = body_start + 8
                while opt + 4 <= body_end:
                    code, olen = struct.unpack_from(endian + "HH", blob, opt)
                    opt += 4
                    if code == 0:
                        break
                    if opt + olen > body_end:
                        break
                    if code == 9 and olen >= 1:  # if_tsresol
                        b = blob[opt]
                        resolution = (2.0 ** -(b & 0x7F)) if (b & 0x80) else (10.0 ** -b)
                    opt += (olen + 3) & ~3
                interfaces.append((linktype, resolution))
                linktypes.add(linktype)
        elif block_type == 6:  # Enhanced Packet Block
            if body_start + 20 <= body_end:
                iface, ts_hi, ts_lo, caplen, _orig = struct.unpack_from(endian + "IIIII", blob, body_start)
                if iface < len(interfaces):
                    linktype, scale = interfaces[iface]
                    pkt_start = body_start + 20
                    if pkt_start + caplen <= body_end:
                        ts = ((ts_hi << 32) | ts_lo) * scale
                        first_ts = min(first_ts, ts)
                        last_ts = max(last_ts, ts)
                        if _packet_is_syn_no_ack(memoryview(blob)[pkt_start:pkt_start+caplen], linktype):
                            syn_times.append(ts)
                        packets += 1
        elif block_type == 2:  # obsolete Packet Block
            if body_start + 20 <= body_end:
                iface, drops, ts_hi, ts_lo, caplen, _orig = struct.unpack_from(endian + "HHIIII", blob, body_start)
                if iface < len(interfaces):
                    linktype, scale = interfaces[iface]
                    pkt_start = body_start + 20
                    if pkt_start + caplen <= body_end:
                        ts = ((ts_hi << 32) | ts_lo) * scale
                        first_ts = min(first_ts, ts)
                        last_ts = max(last_ts, ts)
                        if _packet_is_syn_no_ack(memoryview(blob)[pkt_start:pkt_start+caplen], linktype):
                            syn_times.append(ts)
                        packets += 1
        pos += block_len
    if not packets:
        raise ValueError("No readable enhanced/packet blocks in PCAPNG")
    return syn_times, sorted(linktypes), first_ts, last_ts, packets


def parse_capture(blob: bytes) -> tuple[list[float], list[int], float, float, int]:
    if blob[:4] == b"\x0a\x0d\x0d\x0a":
        return _parse_pcapng(blob)
    return _parse_classic_pcap(blob)


def iter_capture_blobs(archive_path: Path) -> Iterator[tuple[str, bytes]]:
    def walk(zf: zipfile.ZipFile, prefix: str = ""):
        for info in zf.infolist():
            if info.is_dir():
                continue
            name = f"{prefix}{info.filename}"
            low = info.filename.lower()
            try:
                raw = zf.read(info)
            except Exception as exc:
                print(f"WARN: cannot read {archive_path.name}::{name}: {exc}")
                continue
            if low.endswith((".pcap", ".pcapng", ".cap")):
                yield name, raw
            elif low.endswith(".zip"):
                try:
                    with zipfile.ZipFile(io.BytesIO(raw)) as nested:
                        yield from walk(nested, name + "!")
                except zipfile.BadZipFile:
                    print(f"WARN: nested archive is invalid: {archive_path.name}::{name}")
    with zipfile.ZipFile(archive_path) as zf:
        yield from walk(zf)


def bin_capture(capture: Capture, dt: float) -> tuple[np.ndarray, float]:
    n = int(math.floor(capture.duration / dt))
    if n <= 0:
        return np.zeros(0, dtype=float), capture.first_ts
    rel = np.asarray(capture.syn_times, dtype=float) - capture.first_ts
    rel = rel[(rel >= 0) & (rel < n*dt)]
    # PCAP timestamps are commonly stored in microseconds. Floating-point
    # subtraction can turn an exact bin edge (e.g. 1.00 s) into 0.9999999999
    # and shift events one bin left. Snap only values within 1 microsecond of
    # a fixed-width boundary before histogramming.
    nearest_edge = np.rint(rel / dt) * dt
    rel = np.where(np.abs(rel - nearest_edge) <= 1e-6, nearest_edge, rel)
    counts, _ = np.histogram(rel, bins=np.arange(n+1, dtype=float)*dt)
    return counts.astype(float), capture.first_ts


def estimate_phase(counts: np.ndarray, dt: float) -> tuple[int, float, float, int]:
    """Estimate the 200-s cycle start and report per-cycle phase stability.

    A high aggregate correlation alone can hide phase drift, so each complete
    cycle is also aligned independently. The returned consistency is the share
    of cycles whose peak lies within 0.2 s of the aggregate phase.
    """
    cycle_bins = int(round(WINDOW_S / dt))
    n_cycles = len(counts) // cycle_bins
    if n_cycles < 2:
        return 0, 0.0, 0.0, n_cycles
    cycles = counts[:n_cycles*cycle_bins].reshape(n_cycles, cycle_bins)
    folded = cycles.sum(axis=0)
    template = np.zeros(cycle_bins, dtype=float)
    period_bins = max(1, int(round(PERIOD_S / dt)))
    pulse_bins = max(1, int(round(PULSE_S / dt)))
    active_bins = int(round(ACTIVE_S / dt))
    for start in range(0, active_bins, period_bins):
        template[start:min(start+pulse_bins, cycle_bins)] = 1.0

    ft = np.conj(np.fft.fft(template))
    def best_shift(v: np.ndarray) -> tuple[int, float]:
        corr = np.fft.ifft(np.fft.fft(v) * ft).real
        peak = int(np.argmax(corr))
        z = float((corr[peak] - np.median(corr)) / (np.std(corr) + 1e-12))
        return peak, z

    shift, quality = best_shift(folded)
    individual = [best_shift(row)[0] for row in cycles]
    distance = [min((q-shift) % cycle_bins, (shift-q) % cycle_bins) for q in individual]
    consistency = float(np.mean(np.asarray(distance) * dt <= 0.2))
    return shift, quality, consistency, n_cycles


def build_mask(n: int, dt: float, scenario: str, duration: float, rng: np.random.Generator) -> np.ndarray:
    keep = np.ones(n, dtype=bool)
    if scenario == "none" or duration <= 0:
        return keep
    total = min(n, int(round(duration/dt)))
    if total <= 0:
        return keep
    if scenario == "gap_active":
        start = int(round(((ACTIVE_S-duration)/2.0)/dt))
        keep[start:start+total] = False
    elif scenario == "gap_quiet":
        start = int(round((ACTIVE_S+(ACTIVE_S-duration)/2.0)/dt))
        keep[start:start+total] = False
    elif scenario == "gap_boundary":
        start = int(round((ACTIVE_S-duration/2.0)/dt))
        keep[start:start+total] = False
    elif scenario == "fragmented":
        # Ten short blocks, uniformly spread over the 200-s observation window.
        per = total // 10
        remainder = total % 10
        for j in range(10):
            length = per + (1 if j < remainder else 0)
            center = int(round((j + 0.5) * n / 10))
            start = max(0, min(n-length, center-length//2))
            keep[start:start+length] = False
    elif scenario == "random_points":
        keep[rng.choice(n, size=total, replace=False)] = False
    else:
        raise ValueError(scenario)
    return keep


if njit is not None:
    @njit(cache=True)
    def _fast_lomb_power(t: np.ndarray, y: np.ndarray, freq: np.ndarray) -> np.ndarray:
        """Generalized Lomb-Scargle R² with floating mean, compiled by Numba."""
        n = t.size
        out = np.empty(freq.size, dtype=np.float64)
        ym = np.mean(y)
        denom = 0.0
        for i in range(n):
            z = y[i] - ym
            denom += z*z
        denom += 1e-15
        for j in range(freq.size):
            w = 2.0*np.pi*freq[j]
            sc=0.0; ss=0.0; scc=0.0; sss=0.0; scs=0.0
            sy=0.0; syc=0.0; sys=0.0
            for i in range(n):
                c=np.cos(w*t[i]); sn=np.sin(w*t[i]); yi=y[i]
                sc+=c; ss+=sn; scc+=c*c; sss+=sn*sn; scs+=c*sn
                sy+=yi; syc+=yi*c; sys+=yi*sn
            a00=float(n); a01=sc; a02=ss; a11=scc; a12=scs; a22=sss
            b0=sy; b1=syc; b2=sys
            det=a00*(a11*a22-a12*a12)-a01*(a01*a22-a12*a02)+a02*(a01*a12-a11*a02)
            if abs(det)<1e-20:
                out[j]=0.0
                continue
            d0=b0*(a11*a22-a12*a12)-a01*(b1*a22-a12*b2)+a02*(b1*a12-a11*b2)
            d1=a00*(b1*a22-a12*b2)-b0*(a01*a22-a12*a02)+a02*(a01*b2-b1*a02)
            d2=a00*(a11*b2-b1*a12)-a01*(a01*b2-b1*a02)+b0*(a01*a12-a11*a02)
            explained=(d0*b0+d1*b1+d2*b2)/det - (sy*sy/n)
            value=explained/denom
            out[j]=max(0.0,min(1.0,value))
        return out
else:
    _fast_lomb_power = None


def spectral_measure(x: np.ndarray, keep: np.ndarray, dt: float, method: str) -> tuple[float, float]:
    n = len(x)
    idx = np.flatnonzero(keep)
    if len(idx) < max(30, int(0.2*n)):
        return float("nan"), float("nan")
    t_obs = idx * dt
    y_obs = x[idx].astype(float)
    y_obs = y_obs - np.mean(y_obs)
    if method == "lomb_scargle":
        freq = np.linspace(F_SEARCH_MIN, F_SEARCH_MAX, 201)
        if _fast_lomb_power is not None:
            power = _fast_lomb_power(t_obs.astype(np.float64), y_obs.astype(np.float64), freq.astype(np.float64))
        else:
            power = lombscargle(t_obs, y_obs, 2*np.pi*freq,
                                normalize=True, floating_mean=True)
        f_est = float(freq[int(np.argmax(power))])
        score_band = (freq >= F_SCORE_MIN) & (freq <= F_SCORE_MAX)
        score = float(np.max(power[score_band]) / (np.median(power) + 1e-12))
        return score, f_est

    if method == "fft_linear":
        full = np.interp(np.arange(n), idx, y_obs)
    elif method == "fft_zero":
        full = np.zeros(n, dtype=float)
        full[idx] = y_obs
    else:
        raise ValueError(method)
    freq, power = periodogram(full, fs=1.0/dt, window="hann",
                              detrend="constant", scaling="spectrum", nfft=4*n)
    broad = (freq >= F_SEARCH_MIN) & (freq <= F_SEARCH_MAX)
    band = (freq >= F_SCORE_MIN) & (freq <= F_SCORE_MAX)
    f_est = float(freq[np.flatnonzero(broad)[np.argmax(power[broad])]])
    score = float(np.max(power[band]) / (np.median(power[broad]) + 1e-12))
    return score, f_est


def classify_windows(inventory: pd.DataFrame, captures: list[Capture], dt: float, outdir: Path):
    """Create aligned 200-s windows and metadata; persist counts for audit."""
    windows = []
    for c_idx, cap in enumerate(captures):
        if cap.label not in (0, 1):
            continue
        counts, _ = bin_capture(cap, dt)
        if len(counts) < int(WINDOW_S/dt):
            print(f"WARN: {cap.capture_id} shorter than one 200-s window; skipped")
            continue
        if cap.label == 1:
            phase_bin, phase_quality, phase_consistency, phase_cycles = estimate_phase(counts, dt)
            if phase_cycles < 2:
                print(f"WARN: {cap.capture_id} has fewer than two complete 200-s cycles; skipped because phase repetition cannot be verified")
                continue
        else:
            phase_bin, phase_quality, phase_consistency, phase_cycles = 0, float("nan"), float("nan"), 0
        cycle_n = int(round(WINDOW_S/dt))
        starts = range(phase_bin, len(counts)-cycle_n+1, cycle_n)
        used = 0
        for wi, start in enumerate(starts):
            x = counts[start:start+cycle_n].copy()
            if len(x) != cycle_n:
                continue
            pulse_bins = (np.arange(cycle_n) * dt < ACTIVE_S) & ((np.arange(cycle_n) * dt) % PERIOD_S < PULSE_S)
            windows.append({
                "capture_id": cap.capture_id,
                "capture_index": c_idx,
                "window_index": wi,
                "label": int(cap.label),
                "attack_rate": cap.attack_rate,
                "phase_start_s": phase_bin*dt,
                "phase_quality_z": phase_quality,
                "phase_consistency_0p2s": phase_consistency,
                "phase_cycles_checked": phase_cycles,
                "relative_start_s": start*dt,
                "duration_s": WINDOW_S,
                "pulse_bin_count": int(np.sum(pulse_bins)),
                "counts": x,
            })
            used += 1
        print(f"WINDOWS: {cap.capture_id}: label={cap.label}, {used} x 200s, phase={phase_bin*dt:.3f}s, quality={phase_quality:.2f}, stable={phase_consistency:.2f}")
        # A compact binary series file is useful for audit without duplicating the PCAP.
        np.savez_compressed(outdir / f"series_{c_idx:03d}.npz", counts=counts, dt=dt,
                            capture_id=cap.capture_id, label=cap.label,
                            phase_bin=phase_bin, phase_quality=phase_quality,
                            phase_consistency=phase_consistency, phase_cycles=phase_cycles)
    return windows



def _bootstrap_class_values(records: list[dict], label: int, rng: np.random.Generator) -> np.ndarray:
    """Cluster bootstrap by capture; within one capture use circular blocks of 3 windows."""
    grouped = {}
    for r in records:
        if r["label"] == label and np.isfinite(r["score"]):
            grouped.setdefault(r["capture_id"], []).append(r)
    for cid in grouped:
        grouped[cid].sort(key=lambda z: z["window_index"])
    if not grouped:
        return np.asarray([], dtype=float)
    ids = list(grouped)
    sampled = []
    if len(ids) >= 2:
        for pick in rng.choice(ids, size=len(ids), replace=True):
            sampled.extend([z["score"] for z in grouped[str(pick)]])
    else:
        vals = np.asarray([z["score"] for z in grouped[ids[0]]], dtype=float)
        if len(vals) <= 1:
            return vals.copy()
        block_len = min(3, len(vals))
        while len(sampled) < len(vals):
            start = int(rng.integers(0, len(vals)))
            sampled.extend(vals[(start + np.arange(block_len)) % len(vals)].tolist())
        sampled = sampled[:len(vals)]
    return np.asarray(sampled, dtype=float)


def bootstrap_auc_interval(test_records: list[dict], seed: int, n_boot: int = 1000) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n_boot):
        n = _bootstrap_class_values(test_records, 0, rng)
        a = _bootstrap_class_values(test_records, 1, rng)
        if len(n) and len(a):
            # Equivalent to ROC AUC (Mann-Whitney form), but avoids the
            # significant per-replicate overhead of sklearn's full ROC routine.
            diff = a[:, None] - n[None, :]
            values.append(float((np.sum(diff > 0) + 0.5*np.sum(diff == 0)) / diff.size))
    if not values:
        return float("nan"), float("nan")
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def evaluate(windows: list[dict], dt: float, outdir: Path, seed: int = SEED):
    if not windows:
        raise RuntimeError("No labelled 200-s windows. Check archive labels and capture durations.")
    # Chronological blocked split inside each pcap file: first 60% for calibration,
    # last 40% for testing. No 200-s window appears in both partitions.
    per_capture = {}
    for w in windows:
        per_capture.setdefault(w["capture_id"], []).append(w)
    train_ids, test_ids = set(), set()
    for cid, items in per_capture.items():
        items.sort(key=lambda z: z["relative_start_s"])
        cut = max(1, int(math.floor(0.6*len(items))))
        for w in items[:cut]:
            train_ids.add((cid, w["window_index"]))
        for w in items[cut:]:
            test_ids.add((cid, w["window_index"]))
    # Scenario grid: durations affect p=d/200; comparisons at each fixed duration
    # hold p constant across position and fragmentation patterns.
    scenarios = [("none", 0.0)]
    for d in GAP_DURATIONS:
        for s in ("gap_active", "gap_quiet", "gap_boundary", "fragmented", "random_points"):
            scenarios.append((s, d))

    rows, detail = [], []
    for scenario, duration in scenarios:
        for method in METHODS:
            scored = []
            for wi, w in enumerate(windows):
                x = w["counts"]
                rng = np.random.default_rng(seed + wi*1009 + int(duration*100) + sum(map(ord, scenario)))
                keep = build_mask(len(x), dt, scenario, duration, rng)
                score, f_est = spectral_measure(x, keep, dt, method)
                period_err = abs(1.0/f_est - PERIOD_S)/PERIOD_S*100 if f_est > 0 else np.nan
                pulse_bins = (np.arange(len(keep)) * dt < ACTIVE_S) & ((np.arange(len(keep)) * dt) % PERIOD_S < PULSE_S)
                q_missing_pulse = float(np.sum((~keep) & pulse_bins) / max(1, np.sum(pulse_bins)))
                rec = {"capture_id":w["capture_id"], "window_index":w["window_index"],
                       "label":w["label"], "scenario":scenario, "gap_duration_s":duration,
                       "missing_fraction":float(np.mean(~keep)), "fraction_attack_pulse_bins_missing_q":q_missing_pulse, "method":method,
                       "score":score, "estimated_frequency_hz":f_est,
                       "period_error_pct":period_err, "attack_rate":w["attack_rate"],
                       "phase_quality_z":w["phase_quality_z"],
                       "phase_consistency_0p2s":w["phase_consistency_0p2s"]}
                detail.append(rec)
                scored.append(rec)

            tr = [r for r in scored if (r["capture_id"], r["window_index"]) in train_ids and r["label"] == 0 and np.isfinite(r["score"])]
            te = [r for r in scored if (r["capture_id"], r["window_index"]) in test_ids and np.isfinite(r["score"])]
            te0 = np.array([r["score"] for r in te if r["label"] == 0], dtype=float)
            te1 = np.array([r["score"] for r in te if r["label"] == 1], dtype=float)
            tr0 = np.array([r["score"] for r in tr], dtype=float)
            auc = np.nan
            if len(te0) and len(te1):
                yy = np.r_[np.zeros(len(te0), dtype=int), np.ones(len(te1), dtype=int)]
                auc = float(roc_auc_score(yy, np.r_[te0, te1]))
            auc_lo, auc_hi = bootstrap_auc_interval(te, seed + len(rows) + 17, n_boot=1000) if len(te0) and len(te1) else (float("nan"), float("nan"))
            threshold = float(np.quantile(tr0, 0.95, method="higher")) if len(tr0) else float("nan")
            tpr = float(np.mean(te1 > threshold)) if len(te1) and np.isfinite(threshold) else float("nan")
            fpr = float(np.mean(te0 > threshold)) if len(te0) and np.isfinite(threshold) else float("nan")
            attack_errors = [r["period_error_pct"] for r in te if r["label"] == 1 and np.isfinite(r["period_error_pct"])]
            rows.append({
                "scenario":scenario, "gap_duration_s":duration,
                "missing_fraction_nominal":duration/WINDOW_S,
                "method":method, "n_train_benign":len(tr0),
                "n_test_benign":len(te0), "n_test_attack":len(te1),
                "roc_auc_test":auc, "auc_ci95_low":auc_lo, "auc_ci95_high":auc_hi,
                "threshold_train_benign_q95":threshold,
                "detection_rate_tpr":tpr, "false_positive_rate_fpr":fpr,
                "mean_q_missing_attack_pulse_bins":float(np.mean([r["fraction_attack_pulse_bins_missing_q"] for r in te if r["label"] == 1])) if len(te1) else np.nan,
                "median_period_error_pct_attack":float(np.median(attack_errors)) if attack_errors else np.nan,
                "iqr_period_error_pct_attack":float(np.subtract(*np.percentile(attack_errors,[75,25]))) if len(attack_errors)>1 else np.nan,
                "note":"ROC AUC and TPR/FPR evaluated on chronological holdout; gaps simulated on already binned packet counts",
            })
    pd.DataFrame(detail).to_csv(outdir / "main_experiment_windows.csv", index=False)
    result = pd.DataFrame(rows)
    result.to_csv(outdir / "main_experiment_results.csv", index=False)
    return result


def run(data_dir: Path, outdir: Path, dt: float, labels_csv: Optional[Path] = None):
    outdir.mkdir(parents=True, exist_ok=True)
    archives = sorted([p for p in data_dir.glob("*.zip") if p.is_file()])
    if not archives:
        raise FileNotFoundError(f"No .zip archives found in {data_dir}. Expected PCAPs_1.zip etc.")

    captures: list[Capture] = []
    inventory = []
    for archive in archives:
        print(f"READ ARCHIVE: {archive.name}")
        found = 0
        for member, raw in iter_capture_blobs(archive):
            label, rate = infer_label(member)
            label_source = "member_name" if label is not None else None
            if label is None:
                label, rate = infer_label(archive.name + " " + member)
                if label is not None:
                    label_source = "archive_or_member_name"
            try:
                syn_ts, linktypes, first_ts, last_ts, packet_count = parse_capture(raw)
                cap = Capture(str(archive), member, label, rate, first_ts, last_ts,
                              packet_count, np.asarray(syn_ts, dtype=float), ",".join(map(str,linktypes)))
                captures.append(cap)
                found += 1
                inventory.append({
                    "capture_id":cap.capture_id,
                    "archive":archive.name, "member":member, "label":label,
                    "attack_rate":rate, "label_source":label_source,
                    "capture_duration_s":cap.duration,
                    "captured_packets":packet_count, "tcp_syn_without_ack":len(syn_ts),
                    "first_timestamp_unix":first_ts, "last_timestamp_unix":last_ts,
                    "linktypes":cap.linktypes, "file_size_bytes":len(raw),
                })
                print(f"  PCAP {member}: {packet_count:,} packets; SYN(no ACK)={len(syn_ts):,}; duration={cap.duration/60:.1f} min; label={label}")
            except Exception as exc:
                inventory.append({"capture_id":f"{archive.name}::{member}",
                                  "archive":archive.name,"member":member,"label":label,
                                  "attack_rate":rate,"label_source":label_source,
                                  "parse_error":str(exc),"file_size_bytes":len(raw)})
                print(f"  ERROR {member}: {exc}")
        if not found:
            print(f"  WARN: no readable PCAP files found in {archive.name}")

    # Produce a transparent label template even when filenames do not encode classes.
    template_rows = []
    for cap in captures:
        template_rows.append({
            "member":cap.member,
            "capture_id":cap.capture_id,
            "guessed_label":cap.label,
            "guessed_attack_rate":cap.attack_rate,
            "label":cap.label if cap.label in (0,1) else "",
            "attack_rate":cap.attack_rate or "",
        })
    if template_rows:
        template_path = outdir / "labels_template.csv"
        pd.DataFrame(template_rows).to_csv(template_path, index=False)
        print(f"Label template saved to {template_path}; fill label=0 (benign) or 1 (attack) where blank.")

    # Manual mapping takes precedence over filename inference. Use capture_id to avoid
    # ambiguity when the same member name occurs in more than one archive.
    manual_keys_used = set()
    if labels_csv is not None:
        labels_df = pd.read_csv(labels_csv)
        required = {"member", "label"}
        if not required.issubset(labels_df.columns):
            raise ValueError("labels CSV must contain columns: member,label (attack_rate optional)")
        manual = {}
        for _, row in labels_df.iterrows():
            if pd.isna(row["label"]) or str(row["label"]).strip() == "":
                continue
            lab = int(row["label"])
            if lab not in (0,1):
                raise ValueError("manual labels must be 0 (benign) or 1 (attack)")
            key = str(row.get("capture_id", ""))
            if key and key != "nan":
                manual[key] = (lab, str(row.get("attack_rate", "manual")))
            else:
                manual[str(row["member"])] = (lab, str(row.get("attack_rate", "manual")))
        for cap in captures:
            if cap.capture_id in manual:
                cap.label, cap.attack_rate = manual[cap.capture_id]
                manual_keys_used.add(cap.capture_id)
            elif cap.member in manual:
                cap.label, cap.attack_rate = manual[cap.member]
                manual_keys_used.add(cap.member)
        unused = sorted(set(manual) - manual_keys_used)
        if unused:
            print("WARN: label keys not matched to a PCAP member: " + "; ".join(unused[:10]))

    # Refresh inventory after manual mapping so its labels match the actual run.
    labels_by_id = {c.capture_id:(c.label,c.attack_rate) for c in captures}
    for rec in inventory:
        key=rec.get("capture_id")
        if key in labels_by_id:
            rec["label"], rec["attack_rate"] = labels_by_id[key]
            if labels_csv is not None and key in manual_keys_used:
                rec["label_source"] = "manual_labels_csv"
    inv = pd.DataFrame(inventory)
    inv.to_csv(outdir / "pcap_inventory.csv", index=False)

    valid = [c for c in captures if c.label in (0,1)]
    found_labels = {c.label for c in valid}
    if found_labels != {0,1}:
        print("Inventory saved, but a main experiment requires both benign (0) and attack (1) PCAP labels.")
        print("Edit labels_template.csv and rerun with --labels-csv <file>.")
        return None

    windows = classify_windows(inv, valid, dt, outdir)
    # Save phase/alignment diagnostics independently from the window-level table.
    phase_rows = []
    for c_idx, cap in enumerate(valid):
        counts, _ = bin_capture(cap, dt)
        if cap.label == 1:
            phase, quality, consistency, cycles = estimate_phase(counts, dt)
        else:
            phase, quality, consistency, cycles = 0, float("nan"), float("nan"), 0
        phase_rows.append({"capture_id":cap.capture_id,"label":cap.label,
                           "attack_rate":cap.attack_rate,"capture_duration_s":cap.duration,
                           "phase_start_s":phase*dt,"phase_quality_z":quality,
                           "phase_consistency_0p2s":consistency,"phase_cycles_checked":cycles})
    pd.DataFrame(phase_rows).to_csv(outdir / "phase_alignment_diagnostics.csv", index=False)

    results = evaluate(windows, dt, outdir)
    metadata = {
        "dataset":"UTSA-2021 Low-rate DoS Attack",
        "archives_scanned":[p.name for p in archives],
        "capture_count":len(captures), "labelled_capture_count":len(valid),
        "window_count":len(windows), "bin_width_s":dt,
        "window_duration_s":WINDOW_S, "expected_attack_period_s":PERIOD_S,
        "attack_pulse_duration_s":PULSE_S, "active_phase_s":ACTIVE_S,
        "quiet_phase_s":WINDOW_S-ACTIVE_S,
        "random_seed":SEED,
        "limitations":[
            "Phase alignment is estimated from the documented periodic pulse template; inspect phase_quality_z and phase_consistency_0p2s before interpreting the results.",
            "The holdout split is chronological within each capture; it is not capture-level when only one pcap is available per class.",
            "AUC intervals use a capture-cluster bootstrap when several files exist, otherwise circular blocks of three 200-s windows within the file.",
            "Repeated masked variants of one cycle are not independent network captures.",
            "The score tests a known fundamental-frequency region (0.8-1.2 Hz); period estimation searches 0.5-1.5 Hz.",
            "Only SYN packets without ACK are used; IPv4/IPv6 and link-layer support depend on parsed capture formats.",
        ],
    }
    (outdir / "experiment_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("\nRESULTS")
    print(results.to_string(index=False))
    print(f"\nSaved results to {outdir}")
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True,
                        help="Directory containing official UTSA-2021 zip archives")
    parser.add_argument("--out-dir", type=Path, default=Path("main_pcap_results"))
    parser.add_argument("--bin-width", type=float, default=0.1,
                        help="Aggregation interval in seconds; initial default 0.1 s")
    parser.add_argument("--labels-csv", type=Path, default=None,
                        help="Optional manual mapping for ambiguous PCAP names: member,label[,attack_rate]")
    args = parser.parse_args()
    if args.bin_width <= 0 or WINDOW_S/args.bin_width > 2_000_000:
        parser.error("--bin-width must be positive and yield at most 2,000,000 samples per 200-s window")
    run(args.data_dir, args.out_dir, args.bin_width, args.labels_csv)

if __name__ == "__main__":
    main()
