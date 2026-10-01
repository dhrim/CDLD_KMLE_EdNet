"""Paired A/B analysis for the preregistered EdNet metadata augmentation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import os
WORKSPACE = Path(os.environ.get("CDLD_WORKSPACE", Path(__file__).resolve().parents[1] / "work"))

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


BASE = (WORKSPACE / '20260917_cdld_finder_5x10/results/EdNet/full')
AUG = (WORKSPACE / '20260920_representation_study/feature_augmented/results/EdNet/full')
OUT = (WORKSPACE / '20260920_representation_study/results/feature_augmentation')
SEEDS = [42, 43, 44, 45, 46]
BOOTSTRAPS = 5000
BOOTSTRAP_SEED = 20260920


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def weighted_corr(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    keep = np.isfinite(x) & np.isfinite(y) & (w > 0)
    x, y, w = x[keep], y[keep], w[keep].astype(float)
    if len(x) < 2 or w.sum() <= 0:
        return float("nan")
    mx, my = np.average(x, weights=w), np.average(y, weights=w)
    vx = np.average((x - mx) ** 2, weights=w)
    vy = np.average((y - my) ** 2, weights=w)
    if vx <= 0 or vy <= 0:
        return float("nan")
    return float(np.average((x - mx) * (y - my), weights=w) / np.sqrt(vx * vy))


def metric(actual: np.ndarray, probability: np.ndarray, weights: np.ndarray, name: str) -> float:
    actual = actual.ravel()
    probability = np.clip(probability.ravel(), 1e-7, 1 - 1e-7)
    weights = np.repeat(weights, probability.size // weights.size)
    if name == "auc":
        return float(roc_auc_score(actual, probability, sample_weight=weights))
    if name == "log_loss":
        return float(np.average(-(actual * np.log(probability) + (1-actual) * np.log(1-probability)), weights=weights))
    if name == "brier":
        return float(np.average((actual - probability) ** 2, weights=weights))
    if name == "accuracy":
        return float(np.average((probability >= .5) == actual, weights=weights))
    raise KeyError(name)


def load_pair(seed: int, protocol: str):
    a_dir = BASE / f"{protocol}_seed{seed}"
    b_dir = AUG / f"{protocol}_seed{seed}"
    a = np.load(a_dir / "predictions.npz")
    b = np.load(b_dir / "predictions.npz")
    audit = {"seed": seed, "protocol": protocol}
    for key in ["person_ids", "query_items", "actual", "query_row_ids", "support_row_ids", "query_timestamp"]:
        same = np.array_equal(a[key], b[key])
        audit[f"same_{key}"] = bool(same)
        if not same:
            raise RuntimeError(f"A/B split mismatch: seed={seed} protocol={protocol} key={key}")
    audit["same_split_file"] = sha256(a_dir / "split.npz") == sha256(b_dir / "split.npz")
    if not audit["same_split_file"]:
        # Content equality above is the inferential requirement; preserve the byte-level diagnostic.
        with np.load(a_dir / "split.npz") as sa, np.load(b_dir / "split.npz") as sb:
            audit["same_split_content"] = bool(sa.files == sb.files and all(np.array_equal(sa[k], sb[k]) for k in sa.files))
        if not audit["same_split_content"]:
            raise RuntimeError(f"A/B split.npz content mismatch: seed={seed} protocol={protocol}")
    k = -1 if protocol == "warm" else 40
    key = f"probability|CDLD|{k}"
    return a, b, key, audit


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    audit_rows, seed_rows, payload = [], [], {}
    for protocol in ["warm", "cold"]:
        for seed in SEEDS:
            a, b, key, audit = load_pair(seed, protocol)
            audit_rows.append(audit)
            people = a["person_ids"].astype(np.int64)
            actual = a["actual"].astype(float)
            pa, pb = a[key].astype(float), b[key].astype(float)
            ones = np.ones(len(people))
            metrics = ["auc", "log_loss", "brier", "accuracy"] if protocol == "warm" else ["auc"]
            row = {"protocol": protocol, "seed": seed, "k": -1 if protocol == "warm" else 40, "n_students": len(people), "n_pairs": actual.size}
            for name in metrics:
                av, bv = metric(actual, pa, ones, name), metric(actual, pb, ones, name)
                row[f"A_{name}"] = av
                row[f"B_{name}"] = bv
                # Positive is improvement for every contrast.
                row[f"delta_{name}"] = (av - bv) if name in {"log_loss", "brier"} else (bv - av)
            if protocol == "cold":
                observed = actual.mean(axis=1)
                score_a, score_b = pa.mean(axis=1), pb.mean(axis=1)
                row["A_score_correlation"] = weighted_corr(observed, score_a, ones)
                row["B_score_correlation"] = weighted_corr(observed, score_b, ones)
                row["delta_score_correlation"] = row["B_score_correlation"] - row["A_score_correlation"]
            seed_rows.append(row)
            payload[(protocol, seed)] = (people, actual, pa, pb)

    seed_frame = pd.DataFrame(seed_rows)
    seed_frame.to_csv(OUT / "seed_metrics.csv", index=False)
    pd.DataFrame(audit_rows).to_csv(OUT / "split_pairing_audit.csv", index=False)

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    union = np.unique(np.concatenate([payload[("warm", s)][0] for s in SEEDS]))
    metrics = ["auc", "log_loss", "brier", "accuracy"]
    draws = {f"warm_{name}": np.empty(BOOTSTRAPS) for name in metrics}
    draws.update({"cold_auc": np.empty(BOOTSTRAPS), "cold_score_correlation": np.empty(BOOTSTRAPS)})
    indices = {(protocol, seed): np.searchsorted(union, payload[(protocol, seed)][0]) for protocol in ["warm", "cold"] for seed in SEEDS}
    for draw in range(BOOTSTRAPS):
        multiplicity = np.bincount(rng.integers(0, len(union), len(union)), minlength=len(union))
        for protocol in ["warm", "cold"]:
            accumulator = {name: [] for name in (metrics if protocol == "warm" else ["auc", "score_correlation"])}
            for seed in SEEDS:
                people, actual, pa, pb = payload[(protocol, seed)]
                weights = multiplicity[indices[(protocol, seed)]]
                if weights.sum() == 0:
                    continue
                for name in (["auc", "log_loss", "brier", "accuracy"] if protocol == "warm" else ["auc"]):
                    av, bv = metric(actual, pa, weights, name), metric(actual, pb, weights, name)
                    accumulator[name].append((av - bv) if name in {"log_loss", "brier"} else (bv - av))
                if protocol == "cold":
                    observed = actual.mean(axis=1)
                    accumulator["score_correlation"].append(weighted_corr(observed, pb.mean(axis=1), weights) - weighted_corr(observed, pa.mean(axis=1), weights))
            for name, values in accumulator.items():
                draws[f"{protocol}_{name}"][draw] = np.nanmean(values)

    summary = []
    for key, values in draws.items():
        protocol, metric_name = key.split("_", 1)
        point = seed_frame.loc[seed_frame.protocol == protocol, f"delta_{metric_name}"].mean()
        summary.append({
            "protocol": protocol,
            "metric": metric_name,
            "contrast": "B_minus_A_improvement",
            "estimate": float(point),
            "ci_low": float(np.nanquantile(values, .025)),
            "ci_high": float(np.nanquantile(values, .975)),
            "p_improvement_le_0": float((np.sum(values <= 0) + 1) / (np.sum(np.isfinite(values)) + 1)),
            "bootstrap_replicates": BOOTSTRAPS,
        })
    pd.DataFrame(summary).to_csv(OUT / "paired_bootstrap_summary.csv", index=False)
    np.savez_compressed(OUT / "bootstrap_draws.npz", **draws)
    checksums = {p.name: sha256(p) for p in sorted(OUT.iterdir()) if p.is_file()}
    (OUT / "checksums.json").write_text(json.dumps(checksums, indent=2))
    (OUT / "complete.json").write_text(json.dumps({"seeds": SEEDS, "bootstraps": BOOTSTRAPS, "global_student_union_n": int(len(union))}, indent=2))


if __name__ == "__main__":
    main()
