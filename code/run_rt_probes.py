"""Preregistered EdNet held-out-pair response-time probes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import os
WORKSPACE = Path(os.environ.get("CDLD_WORKSPACE", Path(__file__).resolve().parents[1] / "work"))
import time

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler


ROOT = (WORKSPACE / '20260920_representation_study')
SOURCE = (WORKSPACE / '20260917_cdld_finder_5x10/results/EdNet/full')
RESPONSES = (WORKSPACE / '20260914_cdld_reviewer/cache/EdNet/full/responses.npy')
ELAPSED = ROOT / "aux_cache/elapsed_time_ms.npy"
OUTPUT = ROOT / "results/rt_probes"
SEEDS = [42, 43, 44, 45, 46]
ALPHAS = np.logspace(-6, 6, 13)
FOLDS = 5
FOLD_SEED = 20260920
PRIOR_STRENGTH = 10.0
BOOTSTRAP_DRAWS = 5000
BOOTSTRAP_SEED = 20260920


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def save_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".pending")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))
    temporary.replace(path)


def selected_training_rows(train_rows: np.ndarray, people: np.ndarray, data: np.ndarray) -> np.ndarray:
    """Select rows for sorted person IDs without allocating an 80M-row membership mask."""
    all_people = data["person"]
    starts = np.searchsorted(all_people, people, side="left")
    stops = np.searchsorted(all_people, people, side="right")
    blocks = []
    for start, stop in zip(starts, stops):
        left = np.searchsorted(train_rows, start, side="left")
        right = np.searchsorted(train_rows, stop, side="left")
        blocks.append(train_rows[left:right])
    return np.concatenate(blocks)


def smoothed_means(ids: np.ndarray, y: np.ndarray, size: int) -> tuple[np.ndarray, float]:
    global_mean = float(y.mean())
    counts = np.bincount(ids, minlength=size).astype(np.float64)
    sums = np.bincount(ids, weights=y, minlength=size)
    means = (sums + PRIOR_STRENGTH * global_mean) / (counts + PRIOR_STRENGTH)
    return means, global_mean


def make_features(
    train_people: np.ndarray,
    train_items: np.ndarray,
    train_y: np.ndarray,
    target_people: np.ndarray,
    target_items: np.ndarray,
    user_latent: np.ndarray,
    item_latent: np.ndarray,
    target_correct: np.ndarray,
    condition: str,
) -> np.ndarray:
    person_mean, _ = smoothed_means(train_people, train_y, user_latent.shape[0])
    item_mean, _ = smoothed_means(train_items, train_y, item_latent.shape[0])
    columns = [person_mean[target_people, None], item_mean[target_items, None]]
    if condition in {"M1", "M2"}:
        columns.extend([np.asarray(user_latent[target_people]), np.asarray(item_latent[target_items])])
    if condition in {"B2", "M2"}:
        columns.append(target_correct[:, None].astype(np.float64))
    return np.column_stack(columns)


def choose_alpha(
    people: np.ndarray,
    items: np.ndarray,
    correct: np.ndarray,
    y: np.ndarray,
    user_latent: np.ndarray,
    item_latent: np.ndarray,
    condition: str,
) -> tuple[float, list[dict]]:
    folds = KFold(FOLDS, shuffle=True, random_state=FOLD_SEED)
    losses = np.zeros((FOLDS, len(ALPHAS)), dtype=np.float64)
    records = []
    for fold, (fit, valid) in enumerate(folds.split(y)):
        fit_x = make_features(
            people[fit], items[fit], y[fit], people[fit], items[fit],
            user_latent, item_latent, correct[fit], condition,
        )
        valid_x = make_features(
            people[fit], items[fit], y[fit], people[valid], items[valid],
            user_latent, item_latent, correct[valid], condition,
        )
        scaler = StandardScaler().fit(fit_x)
        fit_x, valid_x = scaler.transform(fit_x), scaler.transform(valid_x)
        for index, alpha in enumerate(ALPHAS):
            model = Ridge(alpha=float(alpha), solver="lsqr")
            model.fit(fit_x, y[fit])
            losses[fold, index] = mean_squared_error(y[valid], model.predict(valid_x))
        records.append({"fold": fold, "fit_n": int(len(fit)), "valid_n": int(len(valid))})
    selected = int(np.argmin(losses.mean(axis=0)))
    for record, row in zip(records, losses):
        record["mse_by_alpha"] = {str(float(alpha)): float(loss) for alpha, loss in zip(ALPHAS, row)}
    return float(ALPHAS[selected]), records


def correlations(actual: np.ndarray, predicted: np.ndarray) -> tuple[float, float]:
    pearson = float(pearsonr(actual, predicted).statistic) if np.std(predicted) else 0.0
    spearman = float(spearmanr(actual, predicted).statistic) if np.std(predicted) else 0.0
    return pearson, spearman


def metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    mse = float(mean_squared_error(actual, predicted))
    pearson, spearman = correlations(actual, predicted)
    return {
        "n": int(len(actual)), "mse": mse, "rmse": float(mse ** 0.5),
        "mae": float(mean_absolute_error(actual, predicted)),
        "r2": float(r2_score(actual, predicted)),
        "pearson": pearson, "spearman": spearman,
    }


def run_seed(seed: int, data: np.ndarray, elapsed: np.ndarray) -> tuple[pd.DataFrame, list[dict], list[dict]]:
    source = SOURCE / f"warm_seed{seed}"
    with np.load(source / "split.npz") as split:
        train_rows = split["train_rows"]
        query_rows = split["query_rows"]
        evaluation_people = split["evaluation_people"]
    rows = selected_training_rows(train_rows, evaluation_people, data)
    query = query_rows.ravel()
    if np.intersect1d(rows, query).size:
        raise RuntimeError("RT train/query overlap")
    train_elapsed = np.asarray(elapsed[rows], dtype=np.float64)
    test_elapsed = np.asarray(elapsed[query], dtype=np.float64)
    train_valid = np.isfinite(train_elapsed) & (train_elapsed > 0)
    test_valid = np.isfinite(test_elapsed) & (test_elapsed > 0)
    rows, train_elapsed = rows[train_valid], train_elapsed[train_valid]
    query, test_elapsed = query[test_valid], test_elapsed[test_valid]
    train_seconds, test_seconds = train_elapsed / 1000.0, test_elapsed / 1000.0
    low, high = np.quantile(train_seconds, [0.001, 0.999])
    train_y = np.log1p(np.clip(train_seconds, low, high))
    test_y = np.log1p(np.clip(test_seconds, low, high))
    train_people, train_items = data["person"][rows], data["item"][rows]
    test_people, test_items = data["person"][query], data["item"][query]
    train_correct, test_correct = data["correct"][rows], data["correct"][query]
    user_latent = np.load(source / "CDLD_user_latents.npy", mmap_mode="r")
    item_latent = np.load(source / "CDLD_item_latents.npy", mmap_mode="r")
    result = pd.DataFrame({
        "seed": seed, "row_id": query, "person": test_people, "item": test_items,
        "correct": test_correct, "actual": test_y,
    })
    metric_rows, fold_rows = [], []
    for condition in ["B1", "M1", "B2", "M2"]:
        alpha, folds = choose_alpha(
            train_people, train_items, train_correct, train_y,
            user_latent, item_latent, condition,
        )
        fit_x = make_features(
            train_people, train_items, train_y, train_people, train_items,
            user_latent, item_latent, train_correct, condition,
        )
        test_x = make_features(
            train_people, train_items, train_y, test_people, test_items,
            user_latent, item_latent, test_correct, condition,
        )
        scaler = StandardScaler().fit(fit_x)
        model = Ridge(alpha=alpha, solver="lsqr").fit(scaler.transform(fit_x), train_y)
        prediction = model.predict(scaler.transform(test_x))
        result[f"predicted_{condition}"] = prediction
        metric_rows.append({"seed": seed, "condition": condition, "alpha": alpha, **metrics(test_y, prediction)})
        fold_rows.extend({"seed": seed, "condition": condition, **record} for record in folds)
    for baseline, extended, contrast in [("B1", "M1", "primary"), ("B2", "M2", "explanatory")]:
        delta = (result[f"predicted_{baseline}"] - result["actual"]) ** 2 - (result[f"predicted_{extended}"] - result["actual"]) ** 2
        result[f"squared_error_improvement_{contrast}"] = delta
    audit = {
        "seed": seed, "evaluation_people": int(len(evaluation_people)),
        "rt_train_rows": int(len(train_y)), "rt_test_rows": int(len(test_y)),
        "test_invalid_excluded": int((~test_valid).sum()),
        "winsor_seconds": [float(low), float(high)], "query_overlap": 0,
    }
    return result, metric_rows, fold_rows + [{"audit": audit}]


def union_student_bootstrap(frame: pd.DataFrame, contrast: str) -> dict:
    column = f"squared_error_improvement_{contrast}"
    grouped = frame.groupby(["seed", "person"])[column].agg(["sum", "count"]).reset_index()
    universe = np.sort(grouped["person"].unique())
    position = {person: index for index, person in enumerate(universe)}
    rng = np.random.default_rng(BOOTSTRAP_SEED + (0 if contrast == "primary" else 1))
    draws = np.empty(BOOTSTRAP_DRAWS, dtype=np.float64)
    seed_tables = {}
    for seed in SEEDS:
        table = grouped[grouped["seed"] == seed]
        indices = np.array([position[value] for value in table["person"]], dtype=int)
        seed_tables[seed] = (indices, table["sum"].to_numpy(), table["count"].to_numpy())
    for draw in range(BOOTSTRAP_DRAWS):
        weights = rng.multinomial(len(universe), np.full(len(universe), 1 / len(universe)))
        seed_values = []
        for seed in SEEDS:
            indices, sums, counts = seed_tables[seed]
            denominator = np.dot(weights[indices], counts)
            seed_values.append(float(np.dot(weights[indices], sums) / denominator))
        draws[draw] = np.mean(seed_values)
    observed = float(np.mean([
        group[column].mean() for _, group in frame.groupby("seed", sort=True)
    ]))
    return {
        "contrast": contrast, "delta_mse": observed,
        "ci95": [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))],
        "probability_positive": float(np.mean(draws > 0)),
        "bootstrap_draws": BOOTSTRAP_DRAWS, "global_student_union": int(len(universe)),
    }


def main() -> None:
    auxiliary = json.loads((ROOT / "aux_cache/complete.json").read_text())
    if auxiliary["rows"] != 79_905_206 or "Exact equality" not in auxiliary["alignment"]:
        raise RuntimeError("Auxiliary alignment gate failed")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    data = np.load(RESPONSES, mmap_mode="r")
    elapsed = np.load(ELAPSED, mmap_mode="r")
    if len(data) != len(elapsed):
        raise RuntimeError("Response/elapsed row count mismatch")
    started = time.time()
    frames, metrics_rows, fold_rows = [], [], []
    for seed in SEEDS:
        frame, seed_metrics, seed_folds = run_seed(seed, data, elapsed)
        frame.to_csv(OUTPUT / f"predictions_seed{seed}.csv.gz", index=False, compression="gzip")
        frames.append(frame)
        metrics_rows.extend(seed_metrics)
        fold_rows.extend(seed_folds)
        print(f"completed RT seed={seed}", flush=True)
    combined = pd.concat(frames, ignore_index=True)
    pd.DataFrame(metrics_rows).to_csv(OUTPUT / "metrics.csv", index=False)
    save_json(OUTPUT / "folds_and_audit.json", fold_rows)
    contrasts = [union_student_bootstrap(combined, name) for name in ["primary", "explanatory"]]
    save_json(OUTPUT / "contrasts.json", contrasts)
    manifest = {
        "seeds": SEEDS, "alphas": ALPHAS.tolist(), "folds": FOLDS,
        "fold_seed": FOLD_SEED, "prior_strength": PRIOR_STRENGTH,
        "bootstrap_draws": BOOTSTRAP_DRAWS, "elapsed_seconds": time.time() - started,
        "responses_sha256": auxiliary["reference_response_sha256"],
        "elapsed_sha256": auxiliary["elapsed_time_sha256"],
    }
    save_json(OUTPUT / "complete.json", manifest)
    checksums = {path.name: sha256(path) for path in OUTPUT.iterdir() if path.is_file()}
    save_json(OUTPUT / "checksums.json", checksums)


if __name__ == "__main__":
    main()
