"""Entity-union bootstrap and IRT target-stability analysis for RQ2."""

from __future__ import annotations

import hashlib
import json
from itertools import combinations
from pathlib import Path
import os
WORKSPACE = Path(os.environ.get("CDLD_WORKSPACE", Path(__file__).resolve().parents[1] / "work"))

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


ROOT = (WORKSPACE / '20260920_representation_study')
STATIC = ROOT / "results/static_probes"
SOURCE = (WORKSPACE / '20260917_cdld_finder_5x10/results')
OUT = ROOT / "results/static_inference"
SEEDS = [42, 43, 44, 45, 46]
BOOTSTRAPS = 5000
RNG_SEED = 20260920


def sha256(path: Path) -> str:
    h = hashlib.sha256(path.read_bytes())
    return h.hexdigest()


def weighted_r2(y: np.ndarray, p: np.ndarray, w: np.ndarray) -> float:
    keep = np.isfinite(y) & np.isfinite(p) & (w > 0)
    y, p, w = y[keep], p[keep], w[keep].astype(float)
    if w.sum() <= 0:
        return float("nan")
    center = np.average(y, weights=w)
    total = np.sum(w * (y-center)**2)
    return float(1 - np.sum(w * (y-p)**2) / total) if total > 0 else float("nan")


def weighted_log_loss(y: np.ndarray, probability_columns: np.ndarray, w: np.ndarray) -> float:
    classes = np.sort(np.unique(y))
    lookup = {value: index for index, value in enumerate(classes)}
    selected = np.array([probability_columns[i, lookup[value]] for i, value in enumerate(y)])
    selected = np.clip(selected, 1e-12, 1)
    return float(np.average(-np.log(selected), weights=w))


def bootstrap_contrast(frame: pd.DataFrame, unit: str, dataset: str, target: str, baseline: str, extended: str) -> dict:
    subset = frame[(frame.dataset == dataset) & (frame.unit == unit) & (frame.target == target)]
    a = subset[subset.features == baseline].copy()
    b = subset[subset.features == extended].copy()
    keys = ["dataset", "seed", "unit", "target", "entity_id", "actual"]
    paired = a.merge(b, on=keys, suffixes=("_baseline", "_extended"), validate="one_to_one")
    universe = np.sort(paired.entity_id.unique())
    by_seed = {seed: group for seed, group in paired.groupby("seed", sort=True)}
    index = {seed: np.searchsorted(universe, group.entity_id.to_numpy()) for seed, group in by_seed.items()}
    rng = np.random.default_rng(RNG_SEED + sum(map(ord, dataset + unit + target)))
    draws = np.empty(BOOTSTRAPS)
    for draw in range(BOOTSTRAPS):
        mult = np.bincount(rng.integers(0, len(universe), len(universe)), minlength=len(universe))
        values = []
        for seed, group in by_seed.items():
            w = mult[index[seed]]
            y = group.actual.to_numpy(float)
            if target == "part":
                cols_a = sorted([c for c in group if c.startswith("probability_part_") and c.endswith("_baseline")])
                cols_b = sorted([c for c in group if c.startswith("probability_part_") and c.endswith("_extended")])
                pa, pb = group[cols_a].to_numpy(float), group[cols_b].to_numpy(float)
                values.append(weighted_log_loss(y, pa, w) - weighted_log_loss(y, pb, w))
            else:
                values.append(weighted_r2(y, group.predicted_extended.to_numpy(float), w) - weighted_r2(y, group.predicted_baseline.to_numpy(float), w))
        draws[draw] = np.nanmean(values)
    observed = []
    for _, group in by_seed.items():
        y, w = group.actual.to_numpy(float), np.ones(len(group))
        if target == "part":
            cols_a = sorted([c for c in group if c.startswith("probability_part_") and c.endswith("_baseline")])
            cols_b = sorted([c for c in group if c.startswith("probability_part_") and c.endswith("_extended")])
            observed.append(weighted_log_loss(y, group[cols_a].to_numpy(float), w) - weighted_log_loss(y, group[cols_b].to_numpy(float), w))
        else:
            observed.append(weighted_r2(y, group.predicted_extended.to_numpy(float), w) - weighted_r2(y, group.predicted_baseline.to_numpy(float), w))
    return {
        "dataset": dataset, "unit": unit, "target": target,
        "baseline": baseline, "extended": extended,
        "metric": "delta_log_loss_B1_minus_M1" if target == "part" else "delta_oof_r2_M1_minus_B1",
        "estimate": float(np.mean(observed)),
        "ci_low": float(np.quantile(draws, .025)), "ci_high": float(np.quantile(draws, .975)),
        "p_improvement_le_0": float((np.sum(draws <= 0)+1)/(BOOTSTRAPS+1)),
        "global_entity_union_n": int(len(universe)), "bootstraps": BOOTSTRAPS,
    }, draws


def stability() -> pd.DataFrame:
    rows = []
    for dataset in ["KMLE", "EdNet"]:
        for model, filename, params in [
            ("Rasch", "Rasch_item_parameters.csv", ["b"]),
            ("2PL", "2PL_item_parameters.csv", ["a", "b"]),
        ]:
            tables = {seed: pd.read_csv(SOURCE / dataset / "full" / f"warm_seed{seed}" / filename) for seed in SEEDS}
            for first, second in combinations(SEEDS, 2):
                left, right = tables[first], tables[second]
                valid = left.estimable.to_numpy(bool) & right.estimable.to_numpy(bool)
                for parameter in params:
                    x, y = left[parameter].to_numpy(float)[valid], right[parameter].to_numpy(float)[valid]
                    keep = np.isfinite(x) & np.isfinite(y)
                    x, y = x[keep], y[keep]
                    slope, intercept = np.polyfit(y, x, 1)
                    aligned = intercept + slope*y
                    rows.append({
                        "dataset": dataset, "model": model, "parameter": parameter,
                        "seed_reference": first, "seed_aligned": second, "n_common": len(x),
                        "alignment_intercept": float(intercept), "alignment_slope": float(slope),
                        "pearson": float(pearsonr(x, aligned).statistic),
                        "spearman": float(spearmanr(x, aligned).statistic),
                        "aligned_rmse": float(np.sqrt(np.mean((x-aligned)**2))),
                        "aligned_mae": float(np.mean(np.abs(x-aligned))),
                    })
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(STATIC / "oof_predictions.csv.gz")
    specs = []
    for dataset in ["KMLE", "EdNet"]:
        for target in ["Rasch_theta", "2PL_theta"]:
            specs.append(("student", dataset, target, "response_rate", "latent64"))
        for target in ["Rasch_b", "2PL_b", "2PL_log_a"]:
            specs.append(("item", dataset, target, "response_summaries", "latent64"))
    specs.append(("item", "EdNet", "part", "irt_response_controls", "irt_response_controls_plus_latent64"))
    summaries, draws = [], {}
    for spec in specs:
        result, values = bootstrap_contrast(frame, *spec)
        summaries.append(result)
        draws["|".join(spec[:3])] = values
    pd.DataFrame(summaries).to_csv(OUT / "paired_bootstrap_summary.csv", index=False)
    np.savez_compressed(OUT / "bootstrap_draws.npz", **draws)
    stability().to_csv(OUT / "irt_item_target_stability.csv", index=False)
    checksums = {p.name: sha256(p) for p in OUT.iterdir() if p.is_file()}
    (OUT / "checksums.json").write_text(json.dumps(checksums, indent=2))
    (OUT / "complete.json").write_text(json.dumps({"bootstraps": BOOTSTRAPS, "contrasts": len(specs)}, indent=2))


if __name__ == "__main__":
    main()
