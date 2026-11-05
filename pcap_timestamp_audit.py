#!/usr/bin/env python3
"""Audit packet timestamp continuity and SYN timing in UTSA-2021 ZIP captures.

Run from the directory that contains main_pcap_experiment.py:
  python pcap_timestamp_audit.py --data-dir data --out-dir timestamp_audit

Creates:
  pcap_time_audit.csv       one row per real PCAP
  pcap_window_timeline.csv  one row per complete 200-second interval

The script does NOT run the spectral experiment and does NOT modify PCAP files.
It skips macOS __MACOSX/._* metadata entries automatically.
"""
from __future__ import annotations
import argparse, io, math, struct, zipfile
from pathlib import Path
import numpy as np
import pandas as pd
from main_pcap_experiment import parse_capture

MAGICS = {
    b"\xd4\xc3\xb2\xa1": ("<", 1e-6),
    b"\xa1\xb2\xc3\xd4": (">", 1e-6),
    b"\x4d\x3c\xb2\xa1": ("<", 1e-9),
    b"\xa1\xb2\x3c\x4d": (">", 1e-9),
}

def classic_packet_timestamps(blob: bytes) -> np.ndarray:
    if len(blob) < 24 or blob[:4] not in MAGICS:
        raise ValueError("Not a supported classic PCAP file")
    endian, scale = MAGICS[blob[:4]]
    pos = 24
    out = []
    while pos + 16 <= len(blob):
        sec, frac, incl_len, orig_len = struct.unpack_from(endian + "IIII", blob, pos)
        pos += 16
        if incl_len <= 0 or pos + incl_len > len(blob):
            break
        out.append(sec + frac * scale)
        pos += incl_len
    if not out:
        raise ValueError("No packet records found")
    return np.asarray(out, dtype=np.float64)

def iter_pcaps(zip_path: Path):
    def walk(zf: zipfile.ZipFile, prefix=""):
        for info in zf.infolist():
            if info.is_dir():
                continue
            name = prefix + info.filename
            normalized = name.replace("\\", "/")
            if "__MACOSX/" in normalized or Path(normalized).name.startswith("._"):
                continue
            low = info.filename.lower()
            if low.endswith((".pcap", ".cap")):
                try:
                    yield name, zf.read(info)
                except Exception as e:
                    print(f"WARN read {name}: {e}")
            elif low.endswith(".zip"):
                try:
                    with zipfile.ZipFile(io.BytesIO(zf.read(info))) as nested:
                        yield from walk(nested, name + "!")
                except zipfile.BadZipFile:
                    print(f"WARN invalid nested ZIP: {name}")
    with zipfile.ZipFile(zip_path) as zf:
        yield from walk(zf)

def max_circular_0p1_fraction(syn_rel: np.ndarray):
    if len(syn_rel) == 0:
        return np.nan, np.nan
    phase = np.mod(syn_rel, 1.0)
    hist, _ = np.histogram(phase, bins=100, range=(0.0, 1.0))
    sums = np.array([sum(hist[(i+j) % 100] for j in range(10)) for i in range(100)])
    k = int(np.argmax(sums))
    return float(sums[k] / len(syn_rel)), float(k / 100.0)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=Path("timestamp_audit"))
    ap.add_argument("--window-s", type=float, default=200.0)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    archives = sorted(args.data_dir.glob("*.zip"))
    if not archives:
        raise SystemExit(f"No ZIP archives found in {args.data_dir}")

    file_rows, win_rows = [], []
    for archive in archives:
        print(f"AUDIT {archive.name}")
        for member, raw in iter_pcaps(archive):
            try:
                pkt_ts = classic_packet_timestamps(raw)
                syn_list, linktypes, first_ts, last_ts, packet_count = parse_capture(raw)
                syn_ts = np.asarray(syn_list, dtype=np.float64)
                pkt_sorted = np.sort(pkt_ts)
                dt_order = np.diff(pkt_ts)
                dt_sorted = np.diff(pkt_sorted)
                positive = dt_sorted[dt_sorted > 0]
                span = float(pkt_sorted[-1] - pkt_sorted[0])
                row = {
                    "capture_id": f"{archive.name}::{member}",
                    "archive": archive.name,
                    "member": member,
                    "packet_count": int(len(pkt_ts)),
                    "syn_no_ack_count": int(len(syn_ts)),
                    "timestamp_span_s": span,
                    "timestamp_span_min": span / 60,
                    "out_of_order_adjacent_pairs": int(np.sum(dt_order < 0)),
                    "duplicate_timestamp_adjacent_pairs": int(np.sum(dt_order == 0)),
                    "median_positive_interpacket_gap_s": float(np.median(positive)) if len(positive) else np.nan,
                    "p95_positive_interpacket_gap_s": float(np.quantile(positive, .95)) if len(positive) else np.nan,
                    "p99_positive_interpacket_gap_s": float(np.quantile(positive, .99)) if len(positive) else np.nan,
                    "max_interpacket_gap_s": float(np.max(positive)) if len(positive) else np.nan,
                }
                for threshold in (1, 5, 30, 60, 300, 600):
                    mask = dt_sorted > threshold
                    row[f"gaps_gt_{threshold}s_count"] = int(mask.sum())
                    row[f"gaps_gt_{threshold}s_total_s"] = float(dt_sorted[mask].sum()) if mask.any() else 0.0
                file_rows.append(row)

                # Fixed 200-second timeline from the first observed packet. This is
                # diagnostic only; large empty stretches are not assumed to be genuine monitoring uptime.
                nwin = int(math.floor(span / args.window_s))
                for wi in range(nwin):
                    a = pkt_sorted[0] + wi * args.window_s
                    b = a + args.window_s
                    pa = np.searchsorted(pkt_sorted, a, side="left")
                    pb = np.searchsorted(pkt_sorted, b, side="left")
                    sa = np.searchsorted(syn_ts, a, side="left")
                    sb = np.searchsorted(syn_ts, b, side="left")
                    pseg = pkt_sorted[pa:pb]
                    sseg = syn_ts[sa:sb]
                    syn_rel = sseg - a
                    sec_bins = np.floor(syn_rel).astype(int) if len(syn_rel) else np.array([], dtype=int)
                    tenths = np.floor(syn_rel * 10).astype(int) if len(syn_rel) else np.array([], dtype=int)
                    conc, phase = max_circular_0p1_fraction(syn_rel)
                    gaps = np.diff(pseg) if len(pseg) > 1 else np.array([])
                    win_rows.append({
                        "capture_id": f"{archive.name}::{member}",
                        "window_index": wi,
                        "relative_start_s": wi * args.window_s,
                        "window_duration_s": args.window_s,
                        "total_packets": int(len(pseg)),
                        "syn_no_ack_packets": int(len(sseg)),
                        "syn_packets_per_s": float(len(sseg) / args.window_s),
                        "seconds_with_any_syn": int(len(np.unique(sec_bins))),
                        "0p1s_bins_with_syn": int(len(np.unique(tenths))),
                        "max_0p1s_phase_fraction_of_syn": conc,
                        "dominant_0p1s_phase_start_s_mod_1s": phase,
                        "max_interpacket_gap_inside_window_s": float(np.max(gaps)) if len(gaps) else np.nan,
                    })
                print(f"  {member}: packets={len(pkt_ts):,}, SYN(no ACK)={len(syn_ts):,}, span={span/60:.1f} min, max gap={row['max_interpacket_gap_s']:.1f}s")
            except Exception as e:
                print(f"  SKIP {member}: {e}")

    pd.DataFrame(file_rows).to_csv(args.out_dir / "pcap_time_audit.csv", index=False)
    pd.DataFrame(win_rows).to_csv(args.out_dir / "pcap_window_timeline.csv", index=False)
    print("\nSaved:")
    print(args.out_dir / "pcap_time_audit.csv")
    print(args.out_dir / "pcap_window_timeline.csv")
    if file_rows:
        cols = ["archive", "member", "timestamp_span_min", "max_interpacket_gap_s", "gaps_gt_60s_count", "gaps_gt_60s_total_s", "syn_no_ack_count"]
        print("\nFILE SUMMARY")
        print(pd.DataFrame(file_rows)[cols].to_string(index=False))

if __name__ == "__main__":
    main()
