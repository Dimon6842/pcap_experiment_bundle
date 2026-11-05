#!/usr/bin/env python3
"""Audit PCAP/PCAPNG timestamp continuity and SYN timing in UTSA-2021 ZIP files.

Run from the project directory:
  python pcap_timestamp_audit_v2.py --data-dir data --out-dir timestamp_audit_v2

Outputs:
  pcap_time_audit.csv       one row per successfully parsed capture
  pcap_window_timeline.csv one row per complete 200-second interval
  audit_skipped.csv         captures that could not be parsed, with reason

Supports classic PCAP and PCAPNG, including nested ZIP files. Does not alter
source archives and does not run the spectral experiment.
"""
from __future__ import annotations
import argparse, io, math, struct, zipfile
from pathlib import Path
import numpy as np
import pandas as pd
from main_pcap_experiment import parse_capture

CLASSIC_MAGICS = {
    b"\xd4\xc3\xb2\xa1": ("<", 1e-6),
    b"\xa1\xb2\xc3\xd4": (">", 1e-6),
    b"\x4d\x3c\xb2\xa1": ("<", 1e-9),
    b"\xa1\xb2\x3c\x4d": (">", 1e-9),
}
PCAPNG_SHB = b"\x0a\x0d\x0d\x0a"


def all_packet_timestamps(blob: bytes) -> tuple[np.ndarray, str]:
    """Return packet timestamps in file order and the detected capture format."""
    if blob[:4] in CLASSIC_MAGICS:
        endian, scale = CLASSIC_MAGICS[blob[:4]]
        if len(blob) < 24:
            raise ValueError("Truncated classic PCAP global header")
        pos, out = 24, []
        while pos + 16 <= len(blob):
            sec, frac, incl_len, _orig_len = struct.unpack_from(endian + "IIII", blob, pos)
            pos += 16
            if incl_len <= 0 or pos + incl_len > len(blob):
                break
            out.append(sec + frac * scale)
            pos += incl_len
        if not out:
            raise ValueError("No packet records found in classic PCAP")
        return np.asarray(out, dtype=np.float64), "classic_pcap"

    if blob[:4] == PCAPNG_SHB:
        pos, endian = 0, None
        interfaces: list[float] = []  # seconds per timestamp tick
        out: list[float] = []
        while pos + 12 <= len(blob):
            raw_type = blob[pos:pos+4]
            if raw_type == PCAPNG_SHB:
                if pos + 12 > len(blob):
                    break
                bom = blob[pos+8:pos+12]
                if bom == b"\x4d\x3c\x2b\x1a":
                    endian = "<"
                elif bom == b"\x1a\x2b\x3c\x4d":
                    endian = ">"
                else:
                    raise ValueError("Unknown PCAPNG byte-order magic")
                interfaces = []  # interface IDs reset at each section
                block_type = 0x0A0D0D0A
            else:
                if endian is None:
                    raise ValueError("PCAPNG section header missing")
                block_type = struct.unpack_from(endian + "I", blob, pos)[0]
            block_len = struct.unpack_from(endian + "I", blob, pos+4)[0]
            if block_len < 12 or pos + block_len > len(blob):
                break
            body_start, body_end = pos + 8, pos + block_len - 4
            if block_type == 1:  # Interface Description Block
                if body_start + 8 <= body_end:
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
                    interfaces.append(resolution)
            elif block_type == 6:  # Enhanced Packet Block
                if body_start + 20 <= body_end:
                    iface, ts_hi, ts_lo, caplen, _orig = struct.unpack_from(endian + "IIIII", blob, body_start)
                    if iface < len(interfaces) and body_start + 20 + caplen <= body_end:
                        out.append(((ts_hi << 32) | ts_lo) * interfaces[iface])
            elif block_type == 2:  # obsolete Packet Block
                if body_start + 20 <= body_end:
                    iface, _drops, ts_hi, ts_lo, caplen, _orig = struct.unpack_from(endian + "HHIIII", blob, body_start)
                    if iface < len(interfaces) and body_start + 20 + caplen <= body_end:
                        out.append(((ts_hi << 32) | ts_lo) * interfaces[iface])
            pos += block_len
        if not out:
            raise ValueError("No packet records found in PCAPNG")
        return np.asarray(out, dtype=np.float64), "pcapng"

    raise ValueError(f"Unsupported capture magic: {blob[:4].hex()}")


def iter_pcaps(zip_path: Path):
    def walk(zf: zipfile.ZipFile, prefix=""):
        for info in zf.infolist():
            if info.is_dir():
                continue
            normalized = (prefix + info.filename).replace("\\", "/")
            if "__MACOSX/" in normalized or Path(normalized).name.startswith("._"):
                continue
            low = info.filename.lower()
            if low.endswith((".pcap", ".pcapng", ".cap")):
                try:
                    yield normalized, zf.read(info)
                except Exception as exc:
                    yield normalized, exc
            elif low.endswith(".zip"):
                try:
                    with zipfile.ZipFile(io.BytesIO(zf.read(info))) as nested:
                        yield from walk(nested, normalized + "!")
                except Exception as exc:
                    yield normalized, exc
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
    ap.add_argument("--out-dir", type=Path, default=Path("timestamp_audit_v2"))
    ap.add_argument("--window-s", type=float, default=200.0)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    archives = sorted(p for p in args.data_dir.glob("*.zip") if p.is_file())
    if not archives:
        raise SystemExit(f"No ZIP archives found in {args.data_dir}")
    expected = {"PCAPs_1.zip", "PCAPs_2.zip", "PCAPs_3.zip", "Benign_Only.zip"}
    present = {p.name for p in archives}
    print("ZIP archives found:", ", ".join(p.name for p in archives))
    missing = sorted(expected - present)
    if missing:
        print("WARNING: expected archive(s) not found:", ", ".join(missing))

    file_rows, win_rows, skipped = [], [], []
    for archive in archives:
        print(f"\nAUDIT {archive.name}")
        found, parsed = 0, 0
        for member, raw in iter_pcaps(archive):
            found += 1
            if isinstance(raw, Exception):
                skipped.append({"archive": archive.name, "member": member, "reason": str(raw)})
                print(f"  SKIP {member}: {raw}")
                continue
            try:
                pkt_ts_file_order, fmt = all_packet_timestamps(raw)
                syn_list, _linktypes, _first, _last, _count = parse_capture(raw)
                pkt_sorted = np.sort(pkt_ts_file_order)
                syn_sorted = np.sort(np.asarray(syn_list, dtype=np.float64))  # searchsorted requires sorted input
                dt_order = np.diff(pkt_ts_file_order)
                dt_sorted = np.diff(pkt_sorted)
                positive = dt_sorted[dt_sorted > 0]
                span = float(pkt_sorted[-1] - pkt_sorted[0])
                row = {
                    "capture_id": f"{archive.name}::{member}",
                    "archive": archive.name,
                    "member": member,
                    "pcap_format": fmt,
                    "packet_count": int(len(pkt_ts_file_order)),
                    "syn_no_ack_count": int(len(syn_sorted)),
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
                parsed += 1

                nwin = int(math.floor(span / args.window_s))
                for wi in range(nwin):
                    a = pkt_sorted[0] + wi * args.window_s
                    b = a + args.window_s
                    pa = np.searchsorted(pkt_sorted, a, side="left")
                    pb = np.searchsorted(pkt_sorted, b, side="left")
                    sa = np.searchsorted(syn_sorted, a, side="left")
                    sb = np.searchsorted(syn_sorted, b, side="left")
                    pseg = pkt_sorted[pa:pb]
                    sseg = syn_sorted[sa:sb]
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
                print(f"  OK {member}: {fmt}, packets={len(pkt_ts_file_order):,}, SYN(no ACK)={len(syn_sorted):,}, span={span/60:.1f} min, max gap={row['max_interpacket_gap_s']:.1f}s")
            except Exception as exc:
                skipped.append({"archive": archive.name, "member": member, "reason": str(exc)})
                print(f"  SKIP {member}: {exc}")
        print(f"  SUMMARY: PCAP entries found={found}, successfully audited={parsed}")

    pd.DataFrame(file_rows).to_csv(args.out_dir / "pcap_time_audit.csv", index=False)
    pd.DataFrame(win_rows).to_csv(args.out_dir / "pcap_window_timeline.csv", index=False)
    pd.DataFrame(skipped, columns=["archive", "member", "reason"]).to_csv(args.out_dir / "audit_skipped.csv", index=False)
    print("\nSaved outputs to:", args.out_dir)
    print(f"Successful captures={len(file_rows)}, skipped captures={len(skipped)}, 200-s windows={len(win_rows)}")
    if file_rows:
        cols = ["archive", "member", "pcap_format", "timestamp_span_min", "max_interpacket_gap_s", "gaps_gt_60s_count", "gaps_gt_60s_total_s", "syn_no_ack_count"]
        print("\nFILE SUMMARY\n", pd.DataFrame(file_rows)[cols].to_string(index=False))

if __name__ == "__main__":
    main()
