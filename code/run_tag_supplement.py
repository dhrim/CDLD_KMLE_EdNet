"""Preregistered held-out-item tag decoding supplement."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import os
WORKSPACE = Path(os.environ.get("CDLD_WORKSPACE", Path(__file__).resolve().parents[1] / "work"))

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = (WORKSPACE / '20260920_representation_study')
SOURCE = (WORKSPACE / '20260917_cdld_finder_5x10/results/EdNet/full')
CACHE = (WORKSPACE / '20260914_cdld_reviewer/cache/EdNet/full/responses.npy')
QUESTIONS = (WORKSPACE / '20260914_cdld_reviewer/data/questions.csv')
OUT = ROOT / "results/tag_supplement"
SEEDS = [42, 43, 44, 45, 46]
CS = np.logspace(-4, 4, 9)
FOLDS = 5
FOLD_SEED = 20260920
MIN_TOTAL = 50
MIN_TEST_POSITIVE = 5
N_JOBS = 16


def sha256(path: Path) -> str:
    h = hashlib.sha256(path.read_bytes())
    return h.hexdigest()


def response_summaries(data, train_rows, n_items):
    item = data["item"][train_rows]
    correct = data["correct"][train_rows].astype(float)
    count = np.bincount(item, minlength=n_items)
    total = np.bincount(item, weights=correct, minlength=n_items)
    rate = np.divide(total, count, out=np.full(n_items, correct.mean()), where=count > 0)
    return rate, count


def decode_one(tag_index, tag, y, baseline, extended):
    outer = StratifiedKFold(FOLDS, shuffle=True, random_state=FOLD_SEED)
    splits = list(outer.split(baseline, y))
    if int(y.sum()) < MIN_TOTAL or any(int(y[test].sum()) < MIN_TEST_POSITIVE for _, test in splits):
        return None
    result = {"tag_index": tag_index, "tag": tag, "positive_n": int(y.sum()), "n": len(y)}
    for label, x in [("B1", baseline), ("M1", extended)]:
        probability = np.full(len(y), np.nan)
        selected = []
        for fold, (train, test) in enumerate(splits):
            inner = StratifiedKFold(FOLDS, shuffle=True, random_state=FOLD_SEED + fold + 1)
            model = Pipeline([
                ("scale", StandardScaler()),
                ("logistic", LogisticRegression(solver="lbfgs", penalty="l2", max_iter=1500)),
            ])
            search = GridSearchCV(model, {"logistic__C": CS}, scoring="average_precision", cv=inner, n_jobs=1, refit=True)
            search.fit(x[train], y[train])
            probability[test] = search.predict_proba(x[test])[:, 1]
            selected.append(float(search.best_params_["logistic__C"]))
        result[f"{label}_average_precision"] = float(average_precision_score(y, probability))
        result[f"{label}_probability"] = probability.astype(np.float32)
        result[f"{label}_selected_C"] = selected
    result["delta_average_precision"] = result["M1_average_precision"] - result["B1_average_precision"]
    result["prevalence"] = float(y.mean())
    return result


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    questions = pd.read_csv(QUESTIONS).sort_values("question_id", key=lambda v: v.str[1:].astype(int)).reset_index(drop=True)
    parsed = [str(value).split(";") if pd.notna(value) and str(value) else [] for value in questions["tags"]]
    vocabulary = sorted({tag for tags in parsed for tag in tags})
    tag_matrix = np.zeros((len(questions), len(vocabulary)), dtype=np.uint8)
    lookup = {tag: index for index, tag in enumerate(vocabulary)}
    for row, tags in enumerate(parsed):
        for tag in tags:
            tag_matrix[row, lookup[tag]] = 1
    data = np.load(CACHE, mmap_mode="r")
    metric_rows, prediction_frames, fold_rows = [], [], []
    for seed in SEEDS:
        directory = SOURCE / f"warm_seed{seed}"
        with np.load(directory / "split.npz") as split:
            train_rows = split["train_rows"]
        latent = np.load(directory / "CDLD_item_latents.npy", mmap_mode="r")
        two_pl = pd.read_csv(directory / "2PL_item_parameters.csv")
        rasch = pd.read_csv(directory / "Rasch_item_parameters.csv")
        rate, count = response_summaries(data, train_rows, latent.shape[0])
        valid = rasch.estimable.to_numpy(bool) & two_pl.estimable.to_numpy(bool)
        valid &= np.isfinite(np.column_stack([two_pl.a, two_pl.b, rate, count])).all(axis=1)
        item_ids = np.flatnonzero(valid)
        baseline = np.column_stack([two_pl.a.to_numpy(float), two_pl.b.to_numpy(float), rate, np.log1p(count)])[valid]
        extended = np.column_stack([baseline, np.asarray(latent[item_ids])])
        answers = Parallel(n_jobs=N_JOBS, verbose=10)(
            delayed(decode_one)(j, tag, tag_matrix[valid, j], baseline, extended)
            for j, tag in enumerate(vocabulary)
        )
        answers = [value for value in answers if value is not None]
        for value in answers:
            row = {k: v for k, v in value.items() if not k.endswith("_probability") and not k.endswith("_selected_C")}
            row["seed"] = seed
            metric_rows.append(row)
            prediction_frames.append(pd.DataFrame({
                "seed": seed, "item_id": item_ids, "tag": value["tag"],
                "actual": tag_matrix[valid, value["tag_index"]],
                "probability_B1": value["B1_probability"], "probability_M1": value["M1_probability"],
            }))
            for condition in ["B1", "M1"]:
                for fold, selected_c in enumerate(value[f"{condition}_selected_C"]):
                    fold_rows.append({"seed": seed, "tag": value["tag"], "condition": condition, "outer_fold": fold, "C": selected_c})
        print(f"completed tags seed={seed} eligible={len(answers)}", flush=True)
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(OUT / "tag_metrics.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(OUT / "selected_parameters.csv", index=False)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    predictions.to_csv(OUT / "oof_predictions.csv.gz", index=False, compression="gzip")
    summary = []
    for seed, group in predictions.groupby("seed"):
        tag_ap_b = group.groupby("tag").apply(lambda d: average_precision_score(d.actual, d.probability_B1), include_groups=False)
        tag_ap_m = group.groupby("tag").apply(lambda d: average_precision_score(d.actual, d.probability_M1), include_groups=False)
        summary.append({
            "seed": seed, "eligible_tags": int(group.tag.nunique()),
            "macro_auprc_B1": float(tag_ap_b.mean()), "macro_auprc_M1": float(tag_ap_m.mean()),
            "delta_macro_auprc": float((tag_ap_m-tag_ap_b).mean()),
            "micro_auprc_B1": float(average_precision_score(group.actual, group.probability_B1)),
            "micro_auprc_M1": float(average_precision_score(group.actual, group.probability_M1)),
            "prevalence_baseline": float(group.actual.mean()),
        })
    pd.DataFrame(summary).to_csv(OUT / "summary.csv", index=False)
    checksums = {p.name: sha256(p) for p in OUT.iterdir() if p.is_file()}
    (OUT / "checksums.json").write_text(json.dumps(checksums, indent=2))
    (OUT / "complete.json").write_text(json.dumps({"seeds": SEEDS, "vocabulary_n": len(vocabulary), "minimum_total": MIN_TOTAL, "minimum_test_positive": MIN_TEST_POSITIVE}, indent=2))


if __name__ == "__main__":
    main()
