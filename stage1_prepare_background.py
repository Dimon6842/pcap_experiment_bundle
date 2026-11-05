#!/usr/bin/env python3
"""Prepare real benign packet-count time series for the article experiment.

Run from the project folder:
  python stage1_prepare_background.py --data-dir data --out-dir stage1_background

Requires pcap_timestamp_audit_v2.py and main_pcap_experiment.py beside this file.
This stage only extracts real benign background data; it does not inject anomalies.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from pcap_timestamp_audit_v2 import all_packet_timestamps, iter_pcaps

WINDOW_S = 200.0
BIN_WIDTH_S = 0.01
CONTINUITY_GAP_S = 5.0


def split_segments(ts, max_gap):
    cuts = np.flatnonzero(np.diff(ts) > max_gap) + 1
    return [part for part in np.split(ts, cuts) if len(part) >= 2]


def make_counts(ts, start, duration, dt):
    n = int(round(duration / dt))
    idx = np.floor((ts - start) / dt).astype(np.int64)
    valid = (idx >= 0) & (idx < n)
    return np.bincount(idx[valid], minlength=n).astype(np.int32)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=Path("stage1_background"))
    ap.add_argument("--window-s", type=float, default=WINDOW_S)
    ap.add_argument("--bin-width", type=float, default=BIN_WIDTH_S)
    ap.add_argument("--continuity-gap", type=float, default=CONTINUITY_GAP_S)
    args = ap.parse_args()

    if args.window_s <= 0 or args.bin_width <= 0:
        ap.error("window duration and bin width must be positive")
    n_bins = int(round(args.window_s / args.bin_width))
    if not np.isclose(n_bins * args.bin_width, args.window_s):
        ap.error("window duration must be divisible by bin width")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    archive = args.data_dir / "Benign_Only.zip"
    if not archive.exists():
        raise SystemExit(f"Missing archive: {archive}")

    rows, series, errors = [], [], []
    for member, raw in iter_pcaps(archive):
        low = member.lower()
        if not low.endswith((".pcap", ".pcapng", ".cap")):
            continue
        if "__macosx/" in low or Path(member).name.startswith("._"):
            continue
        try:
            ts, fmt = all_packet_timestamps(raw)
            ts = np.sort(np.asarray(ts, dtype=np.float64))
            if len(ts) < 2:
                raise ValueError("fewer than two packet timestamps")
            segments = split_segments(ts, args.continuity_gap)
            file_windows = 0
            for seg_i, seg in enumerate(segments):
                n_windows = int(np.floor((seg[-1] - seg[0]) / args.window_s))
                for wi in range(n_windows):
                    start = seg[0] + wi * args.window_s
                    stop = start + args.window_s
                    left = np.searchsorted(seg, start, side="left")
                    right = np.searchsorted(seg, stop, side="left")
                    win_ts = seg[left:right]
                    if len(win_ts) < 2:
                        continue
                    counts = make_counts(win_ts, start, args.window_s, args.bin_width)
                    sid = len(series)
                    series.append(counts)
                    file_windows += 1
                    rows.append({
                        "series_id": sid,
                        "capture": Path(member).name,
                        "member": member,
                        "format": fmt,
                        "segment_index": seg_i,
                        "window_index_in_segment": wi,
                        "window_start_from_capture_s": float(start - ts[0]),
                        "window_duration_s": args.window_s,
                        "bin_width_s": args.bin_width,
                        "packet_total": int(counts.sum()),
                        "mean_packets_per_bin": float(np.mean(counts)),
                        "std_packets_per_bin": float(np.std(counts)),
                        "max_packets_per_bin": int(np.max(counts)),
                        "zero_bin_fraction": float(np.mean(counts == 0)),
                        "p95_packets_per_bin": float(np.quantile(counts, .95)),
                        "p99_packets_per_bin": float(np.quantile(counts, .99)),
                        "source_capture_packet_count": int(len(ts)),
                        "source_capture_duration_s": float(ts[-1] - ts[0]),
                    })
            print(f"OK {Path(member).name}: packets={len(ts):,}, "
                  f"segments={len(segments)}, usable_200s_windows={file_windows}")
        except Exception as exc:
            errors.append({"member": member, "error": str(exc)})
            print(f"SKIP {member}: {exc}")

    if not series:
        raise SystemExit("No complete continuous 200-second benign windows extracted.")

    matrix = np.stack(series).astype(np.int32)
    table = pd.DataFrame(rows)
    table.to_csv(args.out_dir / "background_windows.csv", index=False)
    pd.DataFrame(errors, columns=["member", "error"]).to_csv(
        args.out_dir / "background_errors.csv", index=False)
    np.savez_compressed(args.out_dir / "background_series.npz",
                        counts=matrix,
                        window_ids=table["series_id"].to_numpy(dtype=np.int32),
                        capture_names=table["capture"].to_numpy(dtype=str))
    summary = table.groupby("capture", as_index=False).agg(
        windows=("series_id", "count"),
        median_packets_per_bin=("mean_packets_per_bin", "median"),
        median_zero_bin_fraction=("zero_bin_fraction", "median"),
        median_packet_total=("packet_total", "median"),
        median_p99_packets_per_bin=("p99_packets_per_bin", "median"))
    summary.to_csv(args.out_dir / "background_capture_summary.csv", index=False)

    metadata = {
        "source_archive": archive.name,
        "source_type": "real benign PCAP packet timestamps",
        "window_duration_s": args.window_s,
        "bin_width_s": args.bin_width,
        "continuity_gap_s": args.continuity_gap,
        "bins_per_window": n_bins,
        "captures_with_windows": int(table["capture"].nunique()),
        "windows": int(len(table)),
        "errors": int(len(errors)),
        "note": "No synthetic anomaly or missing-data mask has been applied."
    }
    (args.out_dir / "stage1_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8")
    print("\nSTAGE 1 COMPLETE")
    print(json.dumps(metadata, indent=2))
    print("\nCapture summary:")
    print(summary.to_string(index=False))
    print(f"\nOutputs saved to: {args.out_dir}")


if __name__ == "__main__":
    main()
