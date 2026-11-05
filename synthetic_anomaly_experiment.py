#!/usr/bin/env python3
"""
Контролируемый эксперимент для статьи:
искусственная периодическая аномалия добавляется к реальным измерениям
нормального сетевого трафика из Benign_Only.zip.

Скрипт использует парсер PCAP и три спектральных метода из
main_pcap_experiment.py. Исходный скрипт не изменяется.

Запуск из корневой папки проекта:
python synthetic_anomaly_experiment.py --data-dir data --out-dir synthetic_results --bin-width 0.1

По умолчанию к каждому импульсному интервалу добавляется 1, 2 или 4
SYN-пакета. Период импульсов — 1 секунда, длительность импульса — 0.1 с.
Импульсы добавляются в первую половину каждого 200-секундного окна.

Для каждого тестового PCAP порог обнаружения рассчитывается только по
другим PCAP-файлам (leave-one-capture-out). Один и тот же шаблон пропусков
применяется к исходному и изменённому варианту одного окна.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import main_pcap_experiment as core


def read_benign_captures(data_dir: Path):
    archive = data_dir / "Benign_Only.zip"
    if not archive.exists():
        raise FileNotFoundError(f"Не найден архив: {archive}")

    captures = []
    inventory = []
    errors = []

    print(f"ЧТЕНИЕ НОРМАЛЬНОГО ТРАФИКА: {archive.name}")
    for member, raw in core.iter_capture_blobs(archive):
        try:
            syn_ts, linktypes, first_ts, last_ts, packet_count = core.parse_capture(raw)
            cap = core.Capture(
                archive=str(archive),
                member=member,
                label=0,
                attack_rate="benign",
                first_ts=first_ts,
                last_ts=last_ts,
                packet_count=packet_count,
                syn_times=np.asarray(syn_ts, dtype=float),
                linktypes=",".join(map(str, linktypes)),
            )
            captures.append(cap)
            inventory.append({
                "capture_id": cap.capture_id,
                "member": member,
                "capture_duration_s": cap.duration,
                "captured_packets": packet_count,
                "tcp_syn_without_ack": len(syn_ts),
                "linktypes": cap.linktypes,
            })
            print(
                f"  {member}: {packet_count:,} пакетов; "
                f"SYN без ACK={len(syn_ts):,}; "
                f"длительность={cap.duration / 60:.1f} мин"
            )
        except Exception as exc:
            errors.append({"member": member, "error": str(exc)})
            print(f"  ОШИБКА {member}: {exc}")

    if not captures:
        raise RuntimeError("В Benign_Only.zip не удалось прочитать ни одного PCAP.")

    return captures, pd.DataFrame(inventory), pd.DataFrame(errors)


def make_windows(captures, dt: float):
    n_per_window = int(round(core.WINDOW_S / dt))
    if not math.isclose(n_per_window * dt, core.WINDOW_S, rel_tol=0, abs_tol=1e-8):
        raise ValueError("Ширина бина должна делить 200 секунд без остатка.")

    windows = []
    for capture_index, cap in enumerate(captures):
        counts, _ = core.bin_capture(cap, dt)
        n_windows = len(counts) // n_per_window
        used = 0

        for window_index in range(n_windows):
            start = window_index * n_per_window
            x = counts[start:start + n_per_window].astype(float).copy()
            if len(x) != n_per_window:
                continue

            windows.append({
                "capture_id": cap.capture_id,
                "capture_index": capture_index,
                "window_index": window_index,
                "relative_start_s": start * dt,
                "counts": x,
            })
            used += 1

        print(f"ОКНА: {cap.capture_id}: {used} полных окон по 200 с")

    capture_count = len({w["capture_id"] for w in windows})
    if len(windows) < 6:
        raise RuntimeError(
            f"Получено только {len(windows)} окон. Для эксперимента нужно больше данных."
        )
    if capture_count < 2:
        raise RuntimeError(
            "Найден только один пригодный PCAP-файл. "
            "Для проверки с исключением целого файла нужны минимум два."
        )

    return windows


def inject_periodic_anomaly(x: np.ndarray, dt: float, amplitude: int) -> np.ndarray:
    """
    Добавляет известный периодический компонент к реальному фоновому ряду.
    amplitude — число добавляемых SYN-пакетов в каждом импульсном бине.
    """
    y = x.astype(float).copy()
    t = np.arange(len(y), dtype=float) * dt
    active_phase = t < core.ACTIVE_S
    pulse_phase = np.mod(t, core.PERIOD_S) < core.PULSE_S
    y[active_phase & pulse_phase] += amplitude
    return y


def make_mask(window, scenario: str, duration: float, dt: float, seed: int):
    """Детерминированная маска: одинаковая для пары исходный/аномальный ряд."""
    key = (
        f"{window['capture_id']}|{window['window_index']}|"
        f"{scenario}|{duration}|{seed}"
    )
    local_seed = (
        sum((i + 1) * ord(ch) for i, ch in enumerate(key)) + seed
    ) % (2**32)

    rng = np.random.default_rng(local_seed)
    return core.build_mask(
        len(window["counts"]), dt, scenario, duration, rng
    )


def get_scenarios():
    scenarios = [("none", 0.0)]
    for duration in core.GAP_DURATIONS:
        for scenario in (
            "gap_active",
            "gap_quiet",
            "gap_boundary",
            "fragmented",
            "random_points",
        ):
            scenarios.append((scenario, duration))
    return scenarios


def run_experiment(windows, dt, outdir: Path, amplitudes, seed):
    outdir.mkdir(parents=True, exist_ok=True)

    capture_ids = sorted({w["capture_id"] for w in windows})
    scenarios = get_scenarios()

    fold_rows = []
    detail_rows = []

    for fold_number, held_out in enumerate(capture_ids, start=1):
        train_windows = [w for w in windows if w["capture_id"] != held_out]
        test_windows = [w for w in windows if w["capture_id"] == held_out]

        print(
            f"\nПРОВЕРКА {fold_number}/{len(capture_ids)}: "
            f"тестовый PCAP: {held_out}; "
            f"обучающих окон: {len(train_windows)}; "
            f"тестовых окон: {len(test_windows)}"
        )

        for scenario, duration in scenarios:
            train_masks = {
                (w["capture_id"], w["window_index"]):
                    make_mask(w, scenario, duration, dt, seed)
                for w in train_windows
            }
            test_masks = {
                (w["capture_id"], w["window_index"]):
                    make_mask(w, scenario, duration, dt, seed)
                for w in test_windows
            }

            for method in core.METHODS:
                train_scores = []

                # Порог рассчитывается только на исходном нормальном трафике
                # из PCAP-файлов, которые не входят в тестовую часть.
                for w in train_windows:
                    key = (w["capture_id"], w["window_index"])
                    score, _ = core.spectral_measure(
                        w["counts"], train_masks[key], dt, method
                    )
                    if np.isfinite(score):
                        train_scores.append(score)

                threshold = (
                    float(np.quantile(train_scores, 0.95, method="higher"))
                    if train_scores else float("nan")
                )

                for amplitude in amplitudes:
                    negative_scores = []
                    positive_scores = []
                    period_errors = []

                    for w in test_windows:
                        key = (w["capture_id"], w["window_index"])
                        keep = test_masks[key]

                        original = w["counts"]
                        altered = inject_periodic_anomaly(original, dt, amplitude)

                        score0, freq0 = core.spectral_measure(
                            original, keep, dt, method
                        )
                        score1, freq1 = core.spectral_measure(
                            altered, keep, dt, method
                        )

                        negative_scores.append(score0)
                        positive_scores.append(score1)

                        period_error = (
                            abs(1.0 / freq1 - core.PERIOD_S)
                            / core.PERIOD_S * 100
                            if np.isfinite(freq1) and freq1 > 0
                            else float("nan")
                        )
                        period_errors.append(period_error)

                        common = {
                            "fold": fold_number,
                            "held_out_capture": held_out,
                            "capture_id": w["capture_id"],
                            "window_index": w["window_index"],
                            "scenario": scenario,
                            "gap_duration_s": duration,
                            "method": method,
                            "anomaly_packets_per_pulse_bin": amplitude,
                            "threshold_from_training_benign_q95": threshold,
                        }

                        detail_rows.append({
                            **common,
                            "label": 0,
                            "score": score0,
                            "estimated_frequency_hz": freq0,
                            "period_error_pct": np.nan,
                            "detected": (
                                bool(score0 > threshold)
                                if np.isfinite(score0) and np.isfinite(threshold)
                                else None
                            ),
                        })
                        detail_rows.append({
                            **common,
                            "label": 1,
                            "score": score1,
                            "estimated_frequency_hz": freq1,
                            "period_error_pct": period_error,
                            "detected": (
                                bool(score1 > threshold)
                                if np.isfinite(score1) and np.isfinite(threshold)
                                else None
                            ),
                        })

                    n0 = np.asarray(negative_scores, dtype=float)
                    n1 = np.asarray(positive_scores, dtype=float)
                    valid0 = np.isfinite(n0)
                    valid1 = np.isfinite(n1)

                    if np.any(valid0) and np.any(valid1):
                        labels = np.r_[
                            np.zeros(np.sum(valid0), dtype=int),
                            np.ones(np.sum(valid1), dtype=int),
                        ]
                        auc = float(roc_auc_score(
                            labels, np.r_[n0[valid0], n1[valid1]]
                        ))
                    else:
                        auc = float("nan")

                    fpr = (
                        float(np.mean(n0[valid0] > threshold))
                        if np.isfinite(threshold) and np.any(valid0)
                        else float("nan")
                    )
                    tpr = (
                        float(np.mean(n1[valid1] > threshold))
                        if np.isfinite(threshold) and np.any(valid1)
                        else float("nan")
                    )

                    fold_rows.append({
                        "fold": fold_number,
                        "held_out_capture": held_out,
                        "scenario": scenario,
                        "gap_duration_s": duration,
                        "method": method,
                        "anomaly_packets_per_pulse_bin": amplitude,
                        "n_train_benign_windows": len(train_scores),
                        "n_test_benign_windows": int(np.sum(valid0)),
                        "n_test_anomaly_windows": int(np.sum(valid1)),
                        "threshold_train_benign_q95": threshold,
                        "roc_auc": auc,
                        "detection_rate_tpr": tpr,
                        "false_positive_rate_fpr": fpr,
                        "median_period_error_pct": (
                            float(np.nanmedian(period_errors))
                            if np.any(np.isfinite(period_errors))
                            else float("nan")
                        ),
                    })

    fold_df = pd.DataFrame(fold_rows)
    detail_df = pd.DataFrame(detail_rows)

    fold_df.to_csv(outdir / "synthetic_fold_results.csv", index=False)
    detail_df.to_csv(outdir / "synthetic_window_results.csv", index=False)

    group_columns = [
        "scenario",
        "gap_duration_s",
        "method",
        "anomaly_packets_per_pulse_bin",
    ]
    summary = (
        fold_df.groupby(group_columns, dropna=False)
        .agg(
            folds=("fold", "nunique"),
            mean_roc_auc=("roc_auc", "mean"),
            std_roc_auc=("roc_auc", "std"),
            mean_tpr=("detection_rate_tpr", "mean"),
            mean_fpr=("false_positive_rate_fpr", "mean"),
            mean_median_period_error_pct=("median_period_error_pct", "mean"),
            mean_train_benign_windows=("n_train_benign_windows", "mean"),
            mean_test_windows=("n_test_anomaly_windows", "mean"),
        )
        .reset_index()
    )
    summary.to_csv(outdir / "synthetic_summary.csv", index=False)

    metadata = {
        "experiment": (
            "Synthetic periodic anomaly added to real benign PCAP background"
        ),
        "source_archive": "Benign_Only.zip",
        "source_measurement": (
            "timestamps of TCP SYN packets without ACK, parsed from real PCAP files"
        ),
        "bin_width_s": dt,
        "window_duration_s": core.WINDOW_S,
        "anomaly_period_s": core.PERIOD_S,
        "anomaly_pulse_duration_s": core.PULSE_S,
        "anomaly_active_phase_s": core.ACTIVE_S,
        "anomaly_quiet_phase_s": core.WINDOW_S - core.ACTIVE_S,
        "anomaly_amplitudes_added_packets_per_pulse_bin": amplitudes,
        "gap_durations_s": list(core.GAP_DURATIONS),
        "gap_scenarios": [name for name, _ in scenarios],
        "methods": list(core.METHODS),
        "validation": (
            "Leave-one-capture-out: the tested PCAP is excluded from "
            "threshold calibration."
        ),
        "threshold": "95th percentile of scores on training benign windows",
        "masking": (
            "The same missing-observation mask is applied to the original "
            "and anomaly-injected version of each test window."
        ),
        "limitations": [
            "The anomaly is simulated by adding SYN packet counts to real benign traffic.",
            "The simulated pulse pattern is known and aligned to the 200-second window.",
            "Only TCP SYN packets without ACK are analysed.",
            "The number of independent benign PCAP captures is small.",
            "This experiment tests controlled detectability, not detection of an unknown real attack.",
        ],
    }
    (outdir / "synthetic_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\nСВОДНЫЕ РЕЗУЛЬТАТЫ (первые строки):")
    print(summary.head(18).to_string(index=False))
    print(f"\nРезультаты сохранены в: {outdir.resolve()}")
    print("Файлы:")
    print("  synthetic_summary.csv        — сводные результаты")
    print("  synthetic_fold_results.csv   — результаты отдельно по PCAP")
    print("  synthetic_window_results.csv — результаты по каждому окну")
    print("  synthetic_capture_inventory.csv — перечень исходных PCAP")
    print("  synthetic_parse_errors.csv   — ошибки чтения PCAP")
    print("  synthetic_metadata.json      — параметры и ограничения")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("synthetic_results"))
    parser.add_argument("--bin-width", type=float, default=0.1)
    parser.add_argument(
        "--anomaly-levels",
        type=str,
        default="1,2,4",
        help="Число добавляемых SYN-пакетов на импульсный бин, например 1,2,4",
    )
    parser.add_argument("--seed", type=int, default=20261030)
    args = parser.parse_args()

    if args.bin_width <= 0 or not math.isclose(
        core.WINDOW_S / args.bin_width,
        round(core.WINDOW_S / args.bin_width),
        rel_tol=0,
        abs_tol=1e-8,
    ):
        parser.error("--bin-width должна делить 200 секунд без остатка")

    try:
        amplitudes = sorted({
            int(part.strip())
            for part in args.anomaly_levels.split(",")
            if part.strip()
        })
    except ValueError:
        parser.error("--anomaly-levels должен содержать целые числа через запятую")

    if not amplitudes or any(value <= 0 for value in amplitudes):
        parser.error("Все уровни аномалии должны быть положительными целыми числами")

    captures, inventory, errors = read_benign_captures(args.data_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    inventory.to_csv(
        args.out_dir / "synthetic_capture_inventory.csv", index=False
    )
    errors.to_csv(
        args.out_dir / "synthetic_parse_errors.csv", index=False
    )

    windows = make_windows(captures, args.bin_width)
    print(
        f"\nВсего нормальных окон: {len(windows)}; "
        f"PCAP-файлов: {len({w['capture_id'] for w in windows})}; "
        f"уровни добавляемой аномалии: {amplitudes}"
    )

    run_experiment(windows, args.bin_width, args.out_dir, amplitudes, args.seed)


if __name__ == "__main__":
    main()
