"""Run preregistered RQ2 psychometric and EdNet part probes on frozen 5x10 artifacts."""

from __future__ import annotations

import json
from pathlib import Path
import os
WORKSPACE = Path(os.environ.get("CDLD_WORKSPACE", Path(__file__).resolve().parents[1] / "work"))
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    log_loss,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr, spearmanr


SOURCE = (WORKSPACE / '20260917_cdld_finder_5x10/results')
CACHE = (WORKSPACE / '20260914_cdld_reviewer/cache')
QUESTIONS = (WORKSPACE / '20260914_cdld_reviewer/data/questions.csv')
OUTPUT = (WORKSPACE / '20260920_representation_study/results/static_probes')
SEEDS = [42, 43, 44, 45, 46]
RIDGE_ALPHAS = np.logspace(-6, 6, 13)
LOGISTIC_C = np.logspace(-4, 4, 9)
OUTER_FOLDS = 5
INNER_FOLDS = 5
FOLD_SEED = 20260920


def save_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".pending")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))
    temporary.replace(path)


def safe_corr(function, actual: np.ndarray, predicted: np.ndarray) -> float:
    if np.std(actual) == 0 or np.std(predicted) == 0:
        return float("nan")
    return float(function(actual, predicted).statistic)


def nested_ridge(entity_ids: np.ndarray, features: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, list[dict]]:
    order = np.argsort(entity_ids)
    entity_ids, features, target = entity_ids[order], features[order], target[order]
    outer = KFold(OUTER_FOLDS, shuffle=True, random_state=FOLD_SEED)
    prediction = np.full(len(target), np.nan)
    fold_rows = []
    for fold, (train, test) in enumerate(outer.split(entity_ids)):
        pipeline = Pipeline([("scale", StandardScaler()), ("ridge", Ridge())])
        inner = KFold(INNER_FOLDS, shuffle=True, random_state=FOLD_SEED + fold + 1)
        search = GridSearchCV(
            pipeline,
            {"ridge__alpha": RIDGE_ALPHAS},
            scoring="neg_mean_squared_error",
            cv=inner,
            n_jobs=1,
            refit=True,
        )
        search.fit(features[train], target[train])
        prediction[test] = search.predict(features[test])
        fold_rows.append({
            "fold": fold,
            "train_n": int(len(train)),
            "test_n": int(len(test)),
            "alpha": float(search.best_params_["ridge__alpha"]),
        })
    return prediction[np.argsort(order)], fold_rows


def regression_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    return {
        "n": int(len(actual)),
        "r2": float(r2_score(actual, predicted)),
        "rmse": float(mean_squared_error(actual, predicted) ** 0.5),
        "pearson": safe_corr(pearsonr, actual, predicted),
        "spearman": safe_corr(spearmanr, actual, predicted),
    }


def nested_part_decoder(features: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    outer = StratifiedKFold(OUTER_FOLDS, shuffle=True, random_state=FOLD_SEED)
    classes = np.unique(target)
    predicted = np.empty(len(target), dtype=target.dtype)
    probabilities = np.full((len(target), len(classes)), np.nan)
    fold_rows = []
    for fold, (train, test) in enumerate(outer.split(features, target)):
        model = Pipeline([
            ("scale", StandardScaler()),
            ("logistic", LogisticRegression(penalty="l2", solver="lbfgs", max_iter=2000)),
        ])
        inner = StratifiedKFold(INNER_FOLDS, shuffle=True, random_state=FOLD_SEED + fold + 1)
        search = GridSearchCV(
            model,
            {"logistic__C": LOGISTIC_C},
            scoring="neg_log_loss",
            cv=inner,
            n_jobs=1,
            refit=True,
        )
        search.fit(features[train], target[train])
        predicted[test] = search.predict(features[test])
        fold_probability = search.predict_proba(features[test])
        for column, label in enumerate(search.best_estimator_.named_steps["logistic"].classes_):
            probabilities[test, np.flatnonzero(classes == label)[0]] = fold_probability[:, column]
        fold_rows.append({
            "fold": fold,
            "train_n": int(len(train)),
            "test_n": int(len(test)),
            "C": float(search.best_params_["logistic__C"]),
        })
    return predicted, probabilities, fold_rows


def response_summaries(data: np.ndarray, train_rows: np.ndarray, n_people: int, n_items: int):
    people = data["person"][train_rows]
    items = data["item"][train_rows]
    correct = data["correct"][train_rows].astype(np.float64)
    person_count = np.bincount(people, minlength=n_people)
    person_correct = np.bincount(people, weights=correct, minlength=n_people)
    item_count = np.bincount(items, minlength=n_items)
    item_correct = np.bincount(items, weights=correct, minlength=n_items)
    global_rate = float(correct.mean())
    person_rate = np.divide(person_correct, person_count, out=np.full(n_people, global_rate), where=person_count > 0)
    item_rate = np.divide(item_correct, item_count, out=np.full(n_items, global_rate), where=item_count > 0)
    return person_rate, person_count, item_rate, item_count


def run_seed(dataset: str, seed: int, data: np.ndarray, questions: pd.DataFrame | None) -> tuple[list[dict], list[pd.DataFrame], list[dict]]:
    directory = SOURCE / dataset / "full" / f"warm_seed{seed}"
    with np.load(directory / "split.npz") as split:
        train_rows = split["train_rows"]
        evaluation_people = split["evaluation_people"]
    predictions = np.load(directory / "predictions.npz")
    user_latent = np.load(directory / "CDLD_user_latents.npy", mmap_mode="r")
    item_latent = np.load(directory / "CDLD_item_latents.npy", mmap_mode="r")
    n_people, n_items = user_latent.shape[0], item_latent.shape[0]
    person_rate, person_count, item_rate, item_count = response_summaries(
        data, train_rows, n_people, n_items
    )
    metrics, prediction_frames, fold_records = [], [], []

    for irt_model in ["Rasch", "2PL"]:
        target = predictions[f"theta|{irt_model}|-1"]
        ids = evaluation_people.astype(np.int64)
        feature_sets = {
            "response_rate": person_rate[ids, None],
            "latent64": np.asarray(user_latent[ids]),
        }
        for feature_name, features in feature_sets.items():
            estimated, folds = nested_ridge(ids, features, target)
            record = {"dataset": dataset, "seed": seed, "unit": "student", "target": f"{irt_model}_theta", "features": feature_name}
            record.update(regression_metrics(target, estimated))
            metrics.append(record)
            prediction_frames.append(pd.DataFrame({
                "dataset": dataset, "seed": seed, "unit": "student", "target": f"{irt_model}_theta",
                "features": feature_name, "entity_id": ids, "actual": target, "predicted": estimated,
            }))
            fold_records.extend({"dataset":dataset,"seed":seed,"unit":"student","target":f"{irt_model}_theta","features":feature_name,**fold} for fold in folds)

    two_pl = pd.read_csv(directory / "2PL_item_parameters.csv")
    rasch = pd.read_csv(directory / "Rasch_item_parameters.csv")
    item_targets = {
        "Rasch_b": rasch["b"].to_numpy(float),
        "2PL_b": two_pl["b"].to_numpy(float),
        "2PL_log_a": np.log(two_pl["a"].to_numpy(float)),
    }
    estimable = rasch["estimable"].to_numpy(bool) & two_pl["estimable"].to_numpy(bool)
    estimable &= np.isfinite(np.column_stack(list(item_targets.values()))).all(axis=1)
    item_ids = np.flatnonzero(estimable)
    base_features = np.column_stack((item_rate, np.log1p(item_count)))[estimable]
    latent_features = np.asarray(item_latent[item_ids])
    item_feature_sets = {"response_summaries": base_features, "latent64": latent_features}
    for target_name, full_target in item_targets.items():
        target = full_target[estimable]
        for feature_name, features in item_feature_sets.items():
            estimated, folds = nested_ridge(item_ids, features, target)
            record = {"dataset":dataset,"seed":seed,"unit":"item","target":target_name,"features":feature_name}
            record.update(regression_metrics(target, estimated))
            metrics.append(record)
            prediction_frames.append(pd.DataFrame({
                "dataset":dataset,"seed":seed,"unit":"item","target":target_name,"features":feature_name,
                "entity_id":item_ids,"actual":target,"predicted":estimated,
            }))
            fold_records.extend({"dataset":dataset,"seed":seed,"unit":"item","target":target_name,"features":feature_name,**fold} for fold in folds)

    if dataset == "EdNet":
        part = questions["part"].to_numpy(int)
        valid = estimable
        baseline = np.column_stack((
            two_pl["a"].to_numpy(float), two_pl["b"].to_numpy(float), item_rate, np.log1p(item_count)
        ))[valid]
        extended = np.column_stack((baseline, np.asarray(item_latent[np.flatnonzero(valid)])))
        part_ids = np.flatnonzero(valid)
        for feature_name, features in [("irt_response_controls", baseline), ("irt_response_controls_plus_latent64", extended)]:
            label, probability, folds = nested_part_decoder(features, part[valid])
            record = {
                "dataset":dataset,"seed":seed,"unit":"item","target":"part","features":feature_name,
                "n":int(valid.sum()),"log_loss":float(log_loss(part[valid], probability, labels=np.unique(part))),
                "balanced_accuracy":float(balanced_accuracy_score(part[valid], label)),
                "macro_f1":float(f1_score(part[valid], label, average="macro")),
            }
            metrics.append(record)
            frame = pd.DataFrame({
                "dataset":dataset,"seed":seed,"unit":"item","target":"part","features":feature_name,
                "entity_id":part_ids,"actual":part[valid],"predicted":label,
            })
            for index, class_label in enumerate(np.unique(part)):
                frame[f"probability_part_{class_label}"] = probability[:, index]
            prediction_frames.append(frame)
            fold_records.extend({"dataset":dataset,"seed":seed,"unit":"item","target":"part","features":feature_name,**fold} for fold in folds)
    return metrics, prediction_frames, fold_records


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    questions = pd.read_csv(QUESTIONS).sort_values(
        "question_id", key=lambda values: values.str[1:].astype(int)
    ).reset_index(drop=True)
    all_metrics, all_predictions, all_folds = [], [], []
    started = time.time()
    for dataset in ["KMLE", "EdNet"]:
        data = np.load(CACHE / dataset / "full" / "responses.npy", mmap_mode="r")
        for seed in SEEDS:
            metrics, predictions, folds = run_seed(dataset, seed, data, questions if dataset == "EdNet" else None)
            all_metrics.extend(metrics)
            all_predictions.extend(predictions)
            all_folds.extend(folds)
            pd.DataFrame(all_metrics).to_csv(OUTPUT / "metrics.partial.csv", index=False)
            print(f"completed {dataset} seed={seed}", flush=True)
    pd.DataFrame(all_metrics).to_csv(OUTPUT / "metrics.csv", index=False)
    pd.concat(all_predictions, ignore_index=True).to_csv(OUTPUT / "oof_predictions.csv.gz", index=False, compression="gzip")
    pd.DataFrame(all_folds).to_csv(OUTPUT / "fold_parameters.csv", index=False)
    save_json(OUTPUT / "complete.json", {
        "datasets":["KMLE","EdNet"],"seeds":SEEDS,"outer_folds":OUTER_FOLDS,"inner_folds":INNER_FOLDS,
        "ridge_alphas":RIDGE_ALPHAS.tolist(),"logistic_C":LOGISTIC_C.tolist(),
        "elapsed_seconds":time.time()-started,
    })


if __name__ == "__main__":
    main()
