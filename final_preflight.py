#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""One-pass readiness gate for the UTSA-2021 Fourier/gap study.

This is a PRE-FLIGHT experiment only. It does not write paper-ready results.
It validates the four official archives, excludes non-target SlowRead captures,
removes discontinuous capture intervals (predeclared 5 s rule), builds complete
200 s windows, and tests whether the documented 1 s / 0.1 s pulse signature is
visible in aggregate TCP SYN-without-ACK counts.

Usage:
  python final_preflight.py --data-dir data --out-dir final_preflight

The source-IP address is deliberately NOT used: the dataset authors describe
address rewriting/randomization in their processing. IPs are not detector
features and must not be treated as ground truth here.
"""
from __future__ import annotations
import argparse, io, ipaddress, json, math, re, struct, zipfile
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd

EXPECTED = {"Benign_Only.zip", "PCAPs_1.zip", "PCAPs_2.zip", "PCAPs_3.zip"}
GAP_SPLIT_S = 5.0          # fixed before the final run
WINDOW_S = 200.0           # 100 s pulsed phase + 100 s quiet phase
DT = 0.01                  # 10 ms aggregation
PERIOD_S = 1.0
PULSE_S = 0.1
ACTIVE_S = 100.0

CLASSIC_MAGICS = {
    b"\xd4\xc3\xb2\xa1": ("<", 1e-6),
    b"\xa1\xb2\xc3\xd4": (">", 1e-6),
    b"\x4d\x3c\xb2\xa1": ("<", 1e-9),
    b"\xa1\xb2\x3c\x4d": (">", 1e-9),
}
SHB = b"\x0a\x0d\x0d\x0a"


def parse_l3_l4(pkt: memoryview, linktype: int) -> bool:
    """True iff packet is TCP SYN without ACK. No address is retained."""
    n = len(pkt); off = 0; et = None
    if linktype == 1:  # Ethernet
        if n < 14: return False
        et = struct.unpack_from("!H", pkt, 12)[0]; off = 14
        while et in (0x8100, 0x88A8, 0x9100, 0x9200):
            if n < off + 4: return False
            et = struct.unpack_from("!H", pkt, off + 2)[0]; off += 4
    elif linktype == 113:  # Linux cooked v1
        if n < 16: return False
        et = struct.unpack_from("!H", pkt, 14)[0]; off = 16
    elif linktype == 276:  # Linux cooked v2
        if n < 20: return False
        et = struct.unpack_from("!H", pkt, 0)[0]; off = 20
    elif linktype in (101, 228, 229):  # raw IP
        if n < 1: return False
        ver = pkt[0] >> 4
        et = 0x0800 if ver == 4 else (0x86DD if ver == 6 else 0)
    else:
        return False
    try:
        if et == 0x0800:
            if n < off + 20 or (pkt[off] >> 4) != 4: return False
            ihl = (pkt[off] & 0x0F) * 4
            if ihl < 20 or n < off + ihl or pkt[off + 9] != 6: return False
            frag = struct.unpack_from("!H", pkt, off + 6)[0]
            if frag & 0x1FFF: return False
            tcp = off + ihl
        elif et == 0x86DD:
            if n < off + 40 or (pkt[off] >> 4) != 6: return False
            nxt = pkt[off + 6]; tcp = off + 40
            for _ in range(12):
                if nxt == 6: break
                if nxt in (0, 43, 60, 135):
                    if n < tcp + 2: return False
                    nxt, ln = pkt[tcp], (pkt[tcp + 1] + 1) * 8; tcp += ln
                elif nxt == 44:
                    if n < tcp + 8: return False
                    fragoff = struct.unpack_from("!H", pkt, tcp + 2)[0]
                    if fragoff & 0xFFF8: return False
                    nxt = pkt[tcp]; tcp += 8
                elif nxt == 51:
                    if n < tcp + 2: return False
                    nxt, ln = pkt[tcp], (pkt[tcp + 1] + 2) * 4; tcp += ln
                else: return False
            if nxt != 6: return False
        else:
            return False
        if n < tcp + 20: return False
        dataoff = (pkt[tcp + 12] >> 4) * 4
        if dataoff < 20 or n < tcp + dataoff: return False
        flags = pkt[tcp + 13]
        return bool(flags & 0x02) and not bool(flags & 0x10)
    except Exception:
        return False


def parse_classic(blob: bytes):
    magic = blob[:4]
    if magic not in CLASSIC_MAGICS: raise ValueError("unknown classic PCAP magic")
    endian, scale = CLASSIC_MAGICS[magic]
    if len(blob) < 24: raise ValueError("truncated PCAP header")
    linktype = struct.unpack_from(endian + "IHHIIII", blob, 0)[6]
    pos = 24; pkt_ts = []; syn_ts = []; mv = memoryview(blob)
    while pos + 16 <= len(blob):
        sec, frac, caplen, _orig = struct.unpack_from(endian + "IIII", mv, pos); pos += 16
        if caplen <= 0 or pos + caplen > len(blob): break
        ts = sec + frac * scale
        pkt_ts.append(ts)
        if parse_l3_l4(mv[pos:pos+caplen], linktype): syn_ts.append(ts)
        pos += caplen
    if not pkt_ts: raise ValueError("no packet records in PCAP")
    return np.asarray(pkt_ts, dtype=np.float64), np.asarray(syn_ts, dtype=np.float64), "classic_pcap"


def parse_pcapng(blob: bytes):
    pos = 0; endian = None; interfaces = []; pkt_ts = []; syn_ts = []
    while pos + 12 <= len(blob):
        raw = blob[pos:pos+4]
        if raw == SHB:
            bom = blob[pos+8:pos+12]
            if bom == b"\x4d\x3c\x2b\x1a": endian = "<"
            elif bom == b"\x1a\x2b\x3c\x4d": endian = ">"
            else: raise ValueError("bad PCAPNG byte-order magic")
            interfaces = []; block_type = 0x0A0D0D0A
        else:
            if endian is None: raise ValueError("PCAPNG section header missing")
            block_type = struct.unpack_from(endian + "I", blob, pos)[0]
        blen = struct.unpack_from(endian + "I", blob, pos+4)[0]
        if blen < 12 or pos + blen > len(blob): break
        bs, be = pos+8, pos+blen-4
        if block_type == 1 and bs+8 <= be:  # Interface Description Block
            link = struct.unpack_from(endian + "H", blob, bs)[0]
            resolution = 1e-6; offset = 0; op = bs+8
            while op+4 <= be:
                code, olen = struct.unpack_from(endian + "HH", blob, op); op += 4
                if code == 0: break
                if op+olen > be: break
                if code == 9 and olen >= 1:
                    b = blob[op]
                    resolution = (2.0 ** -(b & 0x7F)) if (b & 0x80) else (10.0 ** -b)
                elif code == 14 and olen >= 8:
                    offset = struct.unpack_from(endian + "q", blob, op)[0]
                op += (olen+3) & ~3
            interfaces.append((link, resolution, offset))
        elif block_type == 6 and bs+20 <= be:  # Enhanced Packet Block
            iface, hi, lo, caplen, _orig = struct.unpack_from(endian + "IIIII", blob, bs)
            if iface < len(interfaces):
                link, resolution, offset = interfaces[iface]; ps = bs+20
                if ps+caplen <= be:
                    ts = ((hi << 32) | lo) * resolution + offset
                    pkt_ts.append(ts)
                    if parse_l3_l4(memoryview(blob)[ps:ps+caplen], link): syn_ts.append(ts)
        elif block_type == 2 and bs+20 <= be:  # obsolete Packet Block
            iface, _drops, hi, lo, caplen, _orig = struct.unpack_from(endian + "HHIIII", blob, bs)
            if iface < len(interfaces):
                link, resolution, offset = interfaces[iface]; ps = bs+20
                if ps+caplen <= be:
                    ts = ((hi << 32) | lo) * resolution + offset
                    pkt_ts.append(ts)
                    if parse_l3_l4(memoryview(blob)[ps:ps+caplen], link): syn_ts.append(ts)
        pos += blen
    if not pkt_ts: raise ValueError("no readable packet records in PCAPNG")
    return np.asarray(pkt_ts, dtype=np.float64), np.asarray(syn_ts, dtype=np.float64), "pcapng"


def parse_capture(blob: bytes):
    if blob[:4] in CLASSIC_MAGICS: return parse_classic(blob)
    if blob[:4] == SHB: return parse_pcapng(blob)
    raise ValueError(f"unsupported capture format: {blob[:4].hex()}")


def iter_pcaps(archive: Path):
    def walk(zf: zipfile.ZipFile, prefix=""):
        for info in zf.infolist():
            if info.is_dir(): continue
            name = (prefix + info.filename).replace("\\", "/")
            low = name.lower(); base = Path(name).name
            if "__macosx/" in low or base.startswith("._"): continue
            try: raw = zf.read(info)
            except Exception as exc:
                yield name, None, str(exc); continue
            ext = Path(info.filename).suffix.lower()
            if ext in (".pcap", ".pcapng", ".cap"):
                yield name, raw, None
            elif ext == ".zip":
                try:
                    with zipfile.ZipFile(io.BytesIO(raw)) as nested:
                        yield from walk(nested, name + "!")
                except Exception as exc:
                    yield name, None, f"nested ZIP error: {exc}"
    with zipfile.ZipFile(archive) as zf:
        yield from walk(zf)


def classify_member(member: str):
    base = Path(member.split("!")[-1]).name.lower()
    if "slowread" in base or "slow_read" in base or "slowget" in base: return None, "excluded_slowread", None
    if "client_normal" in base or "benign" in base or "normal" in base: return 0, "filename", "benign"
    m = re.search(r"syn\s*[-_ ]?(\d+)", base)
    if m: return 1, "filename", f"Syn{m.group(1)}"
    return None, "unclassified", None


def continuous_segments(sorted_ts: np.ndarray, threshold: float):
    if len(sorted_ts) < 2: return []
    cuts = np.flatnonzero(np.diff(sorted_ts) > threshold) + 1
    bounds = np.r_[0, cuts, len(sorted_ts)]
    out = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        if b-a >= 2:
            out.append((float(sorted_ts[a]), float(sorted_ts[b-1]), int(b-a)))
    return out


def phase_template(n: int):
    t = np.arange(n) * DT
    return ((t < ACTIVE_S) & ((t % PERIOD_S) < PULSE_S)).astype(float)


def estimate_phase_and_spectrum(x: np.ndarray):
    """Estimate phase by circular matched filter; independent FFT peak estimate."""
    if len(x) != int(round(WINDOW_S/DT)) or not np.any(x):
        return {"phase_s": np.nan, "phase_z": np.nan, "pulse_share": np.nan,
                "pulse_rate_ratio": np.nan, "peak_hz": np.nan, "period_error_pct": np.nan,
                "spectral_score": np.nan}
    n = len(x); temp = phase_template(n)
    xc = x.astype(float) - float(np.mean(x))
    tc = temp - float(np.mean(temp))
    corr = np.fft.irfft(np.fft.rfft(xc) * np.conj(np.fft.rfft(tc)), n=n)
    shift = int(np.argmax(corr))
    # Confirm the orientation using the actual rolled template score.
    candidates = [shift, (-shift) % n]
    best_shift = max(candidates, key=lambda k: float(np.dot(x, np.roll(temp, k))))
    mask = np.roll(temp.astype(bool), best_shift)
    total = float(np.sum(x))
    pulse_sum = float(np.sum(x[mask]))
    pulse_share = pulse_sum / total if total > 0 else np.nan
    pulse_mean = float(np.mean(x[mask]))
    off_mean = float(np.mean(x[~mask]))
    ratio = pulse_mean / (off_mean + 1e-12)

    # Periodogram with Hann taper; frequencies are spaced at 1/200 = 0.005 Hz.
    y = xc * np.hanning(n)
    power = np.abs(np.fft.rfft(y))**2
    freq = np.fft.rfftfreq(n, DT)
    target = (freq >= 0.8) & (freq <= 1.2)
    search = (freq >= 0.5) & (freq <= 1.5)
    side = (freq >= 0.3) & (freq <= 3.0) & (~target)
    if not np.any(target) or not np.any(search) or not np.any(side) or float(np.max(power[search])) <= 1e-20:
        peak_hz = np.nan; perr = np.nan; spec = np.nan
    else:
        ix = np.flatnonzero(search)[int(np.argmax(power[search]))]
        peak_hz = float(freq[ix])
        perr = abs((1.0/peak_hz)-PERIOD_S)/PERIOD_S*100 if peak_hz > 0 else np.nan
        spec = float(np.max(power[target]) / (np.median(power[side]) + 1e-20))
    z = float((corr[best_shift] - np.median(corr)) / (np.std(corr) + 1e-12))
    return {"phase_s": best_shift*DT, "phase_z": z, "pulse_share": pulse_share,
            "pulse_rate_ratio": ratio, "peak_hz": peak_hz, "period_error_pct": perr,
            "spectral_score": spec}


def make_window_counts(syn_ts: np.ndarray, start: float):
    n = int(round(WINDOW_S/DT)); end = start + WINDOW_S
    a = np.searchsorted(syn_ts, start, side="left")
    b = np.searchsorted(syn_ts, end, side="left")
    rel = syn_ts[a:b] - start
    # Snap timestamp rounding noise only if within 1 microsecond of a bin edge.
    edge = np.rint(rel/DT)*DT
    rel = np.where(np.abs(rel-edge) <= 1e-6, edge, rel)
    counts, _ = np.histogram(rel, bins=np.arange(n+1, dtype=float)*DT)
    return counts.astype(float), int(b-a)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=Path("final_preflight"))
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    archives = sorted(p for p in args.data_dir.glob("*.zip") if p.is_file())
    found = {p.name for p in archives}
    missing = sorted(EXPECTED - found)
    print("EXPECTED ARCHIVES:", ", ".join(sorted(EXPECTED)))
    print("FOUND ARCHIVES:   ", ", ".join(p.name for p in archives) if archives else "none")

    captures = []; parse_errors = []; excluded = []
    for archive in archives:
        print(f"\nSCAN {archive.name}")
        for member, blob, err in iter_pcaps(archive):
            if err:
                parse_errors.append({"archive": archive.name, "member": member, "error": err}); continue
            label, label_source, rate = classify_member(member)
            if label is None:
                excluded.append({"archive": archive.name, "member": member, "reason": label_source})
                print(f"  EXCLUDED {member} ({label_source})")
                continue
            try:
                pkt_ts, syn_ts, fmt = parse_capture(blob)
                file_order = pkt_ts.copy()
                out_of_order = int(np.sum(np.diff(file_order) < 0))
                pkt_ts = np.sort(pkt_ts)
                syn_ts = np.sort(syn_ts)
                if len(pkt_ts) < 2: raise ValueError("fewer than two packet timestamps")
                gaps = np.diff(pkt_ts)
                segs = continuous_segments(pkt_ts, GAP_SPLIT_S)
                eligible = [s for s in segs if (s[1]-s[0]) >= WINDOW_S]
                win_count = 0
                for si, (s0, s1, packet_count) in enumerate(eligible):
                    n_full = int(math.floor((s1-s0)/WINDOW_S))
                    for wi in range(n_full):
                        wstart = s0 + wi*WINDOW_S
                        counts, syn_count = make_window_counts(syn_ts, wstart)
                        if counts.size != int(round(WINDOW_S/DT)): continue
                        # Compute the same spectrum for both classes. For attack windows,
                        # the matched-filter phase fields are used only as a pre-flight check.
                        m = estimate_phase_and_spectrum(counts)
                        captures.append({
                            "capture_id": f"{archive.name}::{member}", "archive": archive.name,
                            "member": member, "label": label, "attack_rate": rate,
                            "format": fmt, "segment_index": si, "window_index_in_segment": wi,
                            "window_start_unix": wstart, "window_duration_s": WINDOW_S,
                            "packet_count_in_segment": packet_count, "syn_count_in_window": syn_count,
                            **m, "phase_shift_s": m["phase_s"]
                        })
                        win_count += 1
                captures.append({
                    "_inventory": True, "capture_id": f"{archive.name}::{member}",
                    "archive": archive.name, "member": member, "label": label,
                    "attack_rate": rate, "format": fmt, "packet_count": len(pkt_ts),
                    "syn_count": len(syn_ts), "timestamp_span_s": float(pkt_ts[-1]-pkt_ts[0]),
                    "max_gap_s": float(np.max(gaps)) if len(gaps) else np.nan,
                    "gaps_gt_5s": int(np.sum(gaps > GAP_SPLIT_S)),
                    "p99_gap_s": float(np.quantile(gaps[gaps > 0], .99)) if np.any(gaps > 0) else np.nan,
                    "out_of_order_in_file": out_of_order, "continuous_segments": len(segs),
                    "segments_ge_200s": len(eligible), "eligible_200s_windows": win_count
                })
                print(f"  OK {member}: {fmt}; packets={len(pkt_ts):,}; SYN={len(syn_ts):,}; segments>={WINDOW_S:g}s={len(eligible)}; windows={win_count}")
            except Exception as exc:
                parse_errors.append({"archive": archive.name, "member": member, "error": str(exc)})
                print(f"  ERROR {member}: {exc}")

    inventory_rows = [x for x in captures if x.get("_inventory")]
    window_rows = [x for x in captures if not x.get("_inventory")]
    inventory = pd.DataFrame(inventory_rows).drop(columns=["_inventory"], errors="ignore")
    windows = pd.DataFrame(window_rows)
    inventory.to_csv(args.out_dir/"preflight_capture_summary.csv", index=False)
    windows.to_csv(args.out_dir/"preflight_windows.csv", index=False)
    pd.DataFrame(parse_errors, columns=["archive","member","error"]).to_csv(args.out_dir/"preflight_errors.csv", index=False)
    pd.DataFrame(excluded, columns=["archive","member","reason"]).to_csv(args.out_dir/"preflight_excluded.csv", index=False)

    # Capture-level pulse evidence: same phase should recur in consecutive 200-s windows.
    pulse_capture_rows = []
    if len(windows):
        attacks = windows[windows.label == 1]
        for cid, g in attacks.groupby("capture_id"):
            vals = g.phase_s.to_numpy(float)
            vals = vals[np.isfinite(vals)]
            if len(vals):
                # Compare phase in absolute capture time, not relative to each
                # segment/window. This remains valid even after a long gap is split.
                vals = (g.window_start_unix.to_numpy(float) + g.phase_s.to_numpy(float)) % WINDOW_S
                ang = vals * (2*np.pi/WINDOW_S)
                mean_ang = np.angle(np.mean(np.exp(1j*ang)))
                center = (mean_ang * WINDOW_S / (2*np.pi)) % WINDOW_S
                dist = np.abs(((vals-center+WINDOW_S/2) % WINDOW_S)-WINDOW_S/2)
                stability = float(np.mean(dist <= 0.2))
            else:
                center, stability = np.nan, np.nan
            pulse_capture_rows.append({
                "capture_id": cid, "windows": len(g),
                "median_pulse_share": float(g.pulse_share.median()),
                "median_pulse_rate_ratio": float(g.pulse_rate_ratio.median()),
                "median_phase_z": float(g.phase_z.median()),
                "median_peak_hz": float(g.peak_hz.median()),
                "median_period_error_pct": float(g.period_error_pct.median()),
                "phase_center_s": center, "phase_stability_within_0p2s": stability,
                "share_windows_pulse_share_ge_0p10": float(np.mean(g.pulse_share >= 0.10)),
                "share_windows_peak_0p8_1p2_hz": float(np.mean(g.peak_hz.between(0.8,1.2)))
            })
    pulse_df = pd.DataFrame(pulse_capture_rows)
    pulse_df.to_csv(args.out_dir/"preflight_attack_signature.csv", index=False)

    reasons = []
    if missing: reasons.append("Missing required ZIP archive(s): " + ", ".join(missing))
    benign_caps = inventory[inventory.label == 0] if len(inventory) else pd.DataFrame()
    attack_caps = inventory[inventory.label == 1] if len(inventory) else pd.DataFrame()
    if len(benign_caps) < 2: reasons.append("Fewer than 2 readable benign PCAP captures")
    if len(attack_caps) < 4: reasons.append("Fewer than 4 readable SYN-attack PCAP captures")
    if len(inventory) and len(parse_errors): reasons.append(f"{len(parse_errors)} PCAP parse/read error(s)")
    benign_w = windows[windows.label == 0] if len(windows) else pd.DataFrame()
    attack_w = windows[windows.label == 1] if len(windows) else pd.DataFrame()
    benign_ids = benign_w.capture_id.nunique() if len(benign_w) else 0
    attack_ids = attack_w.capture_id.nunique() if len(attack_w) else 0
    if len(benign_w) < 20 or benign_ids < 2:
        reasons.append(f"Insufficient continuous benign windows: {len(benign_w)} windows across {benign_ids} captures; need >=20 across >=2")
    if len(attack_w) < 30 or attack_ids < 3:
        reasons.append(f"Insufficient continuous attack windows: {len(attack_w)} windows across {attack_ids} captures; need >=30 across >=3")
    qualified = pulse_df[
        (pulse_df.windows >= 5) &
        (pulse_df.median_pulse_share >= 0.10) &
        (pulse_df.phase_stability_within_0p2s >= 0.70)
    ] if len(pulse_df) else pd.DataFrame()
    if len(qualified) < 3:
        reasons.append(f"Documented pulse pattern confirmed in only {len(qualified)} attack captures; need >=3 captures with >=5 windows each, median pulse-bin share >=0.10 and phase stability >=0.70")

    status = "READY" if not reasons else "BLOCKED"
    report = []
    report += ["UTSA-2021 FINAL PRE-FLIGHT", "="*42, f"STATUS: {status}", ""]
    report += [f"Required archives present: {len(EXPECTED)-len(missing)}/4",
               f"Readable target captures: {len(inventory)}",
               f"Benign captures: {len(benign_caps)}; attack captures: {len(attack_caps)}",
               f"Continuous 200-s windows: benign={len(benign_w)}, attack={len(attack_w)}",
               f"Independent captures contributing windows: benign={benign_ids}, attack={attack_ids}",
               f"Attack captures confirming pulse signature: {len(qualified)}",
               f"Continuous-segment rule: split if adjacent packet gap > {GAP_SPLIT_S:.1f} s",
               f"Window duration: {WINDOW_S:.0f} s; bin width: {DT:.2f} s; expected pulse: {PULSE_S:.1f} s each {PERIOD_S:.0f} s during first {ACTIVE_S:.0f} s of cycle",
               "Source IP addresses are not used for labels, phase detection or detector features.",
               "The 5-s split threshold is fixed before the main run; no 200-s window crosses a >5-s packet gap.", ""]
    if reasons:
        report.append("BLOCKERS:")
        report.extend([f"- {r}" for r in reasons])
        report += ["", "Do NOT run the main experiment. This report is the definitive pre-flight result; do not tune thresholds after seeing test outcomes."]
    else:
        report += ["PRE-FLIGHT PASSED.",
                   "The raw capture set, continuous windows and documented pulse signature meet the predeclared readiness criteria.",
                   "The next step is the single main experiment using the same 5-s continuity rule and 10-ms bins.",
                   "Pre-flight metrics are quality-control checks only and are not inserted into the article."]
    (args.out_dir/"PREFLIGHT_REPORT.txt").write_text("\n".join(report), encoding="utf-8")
    manifest = {
        "status": status, "missing_archives": missing, "parse_errors": parse_errors,
        "capture_counts": {"benign": int(len(benign_caps)), "attack": int(len(attack_caps))},
        "window_counts": {"benign": int(len(benign_w)), "attack": int(len(attack_w))},
        "window_capture_counts": {"benign": int(benign_ids), "attack": int(attack_ids)},
        "qualified_attack_captures": int(len(qualified)),
        "gap_split_threshold_s": GAP_SPLIT_S, "window_s": WINDOW_S, "bin_width_s": DT,
        "blockers": reasons
    }
    (args.out_dir/"preflight_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("\n" + "\n".join(report))
    print(f"\nSaved report and CSVs to: {args.out_dir}")
    print("Zip this folder and send it as ONE attachment.")
    return 0 if status == "READY" else 2

if __name__ == "__main__":
    raise SystemExit(main())
