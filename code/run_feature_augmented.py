
# CELL 2
COMMON_CODE_SHA256 = "88e6286bc435766557e9730555dced35f7b175a6f4702e84c41bd28748be1e12"


# CELL 4
# Optional: run once in a clean kernel, then restart it before continuing.
INSTALL_DEPENDENCIES = False
import sys
import subprocess
import platform
required_packages = [
    "tensorflow[and-cuda]==2.21.0" if platform.system() == "Linux" else "tensorflow==2.21.0",
    "keras==3.15.1", "numpy==2.4.6", "scipy==1.17.1", "scikit-learn==1.9.1",
    "pandas==3.0.5", "matplotlib==3.11.2",
]
if INSTALL_DEPENDENCIES:
    subprocess.check_call([sys.executable, "-m", "pip", "install", *required_packages])
print("Install only if needed; use the pinned versions for the reproducibility run.")


# CELL 7
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "4")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
import gc
import sys
import json
import time
import hashlib
import zipfile
import traceback
from pathlib import Path
import os
WORKSPACE = Path(os.environ.get("CDLD_WORKSPACE", Path(__file__).resolve().parents[1] / "work"))
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.special import expit, logsumexp
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss, cohen_kappa_score
from IPython.display import display

print("Python", sys.version.split()[0])


# CELL 9
DATASET_NAME = "EdNet"
WORK_DIR = Path(os.environ.get("REPRESENTATION_WORK_DIR", Path.cwd()))
NOTEBOOK_PATH = WORK_DIR / "02_ednet_cdld_irt_01.ipynb"
RUN_SEEDS = [int(value) for value in os.environ.get("RUN_SEEDS", "42,43,44,45,46").split(",") if value]
RAW_DATA = WORK_DIR / "data" / "EdNet-KT1.zip"
QUESTIONS_FILE = WORK_DIR / "data" / "questions.csv"


# CELL 11
# Identical scientific settings in the KMLE and EdNet notebooks.
RUN_MODE = "full"                 # "smoke" is a pipeline check, never a paper result.
RUN_EXPERIMENT = True             # False: read saved results without training.
RESUME = True
CONFIG = {
    "seeds": [42, 43, 44, 45, 46],
    "protocols": ["warm", "cold"],
    "evaluation_students": 500,
    "support_counts": [0, 4, 8, 12, 16, 20, 40],
    "query_count": 40,
    "validation_fraction": 0.1,
    "latent_size": 64,
    "batch_size": 8192,
    "finder_cycles": 5,
    "finder_subepochs": 10,
    "predictor_epochs": 25,
    "adaptation_epochs": 300,
    "learning_rate": 0.0005,
    "irt_nodes": 41,
    "irt_max_iterations": 3000,
    "irt_tolerance": 1e-5,
    "irt_person_batch": 4096,
    "bootstrap_draws": 2000,
    "bootstrap_seed": 20260914,
    "correlation_margin": 0.010,
}
if RUN_MODE == "smoke":
    CONFIG.update(seeds=[42], evaluation_students=8, support_counts=[0, 4, 8],
                  query_count=8, finder_cycles=1, finder_subepochs=1,
                  predictor_epochs=1, adaptation_epochs=2, bootstrap_draws=100)
else:
    assert RUN_MODE == "full"
OUTPUT_DIR = WORK_DIR / "results" / DATASET_NAME / RUN_MODE
CACHE_DIR = WORK_DIR / "cache" / DATASET_NAME / RUN_MODE
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)
display(pd.DataFrame({"setting": CONFIG.keys(), "value": [str(v) for v in CONFIG.values()]}))
print("Results:", OUTPUT_DIR)


# CELL 13
RESPONSE_DTYPE = np.dtype([("person", "<i4"), ("item", "<i4"),
                           ("correct", "u1"), ("timestamp", "<i8")])

def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def save_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=True))
    temporary.replace(path)

def save_npz(path, **values):
    path = Path(path)
    temporary = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(temporary, **values)
    temporary.replace(path)

def finish_cache(parts, people, items, description):
    counts = [len(np.load(path, mmap_mode="r")) for path in parts]
    target = CACHE_DIR / "responses.npy"
    temporary = CACHE_DIR / "responses.pending.npy"
    responses = np.lib.format.open_memmap(temporary, mode="w+", dtype=RESPONSE_DTYPE,
                                         shape=(sum(counts),))
    start = 0
    for path, count in zip(parts, counts):
        responses[start:start + count] = np.load(path, mmap_mode="r")
        start += count
    responses.flush()
    del responses
    temporary.replace(target)
    people.to_csv(CACHE_DIR / "people.csv", index=False)
    items.to_csv(CACHE_DIR / "items.csv", index=False)
    description.update(rows=sum(counts), people=len(people), items=len(items),
                       response_sha256=file_sha256(target))
    save_json(CACHE_DIR / "complete.json", description)
    return np.load(target, mmap_mode="r"), description

def load_cache(expected_sources):
    marker = CACHE_DIR / "complete.json"
    if not marker.exists():
        return None
    description = json.loads(marker.read_text())
    if description["source_hashes"] != expected_sources:
        raise RuntimeError("The input changed. Use a new cache directory; do not mix datasets.")
    if file_sha256(CACHE_DIR / "responses.npy") != description["response_sha256"]:
        raise RuntimeError("The response cache checksum failed.")
    print("Verified cached responses:", description["rows"])
    return np.load(CACHE_DIR / "responses.npy", mmap_mode="r"), description


# CELL 15
def prepare_data():
    if not RAW_DATA.is_file() or not QUESTIONS_FILE.is_file():
        raise FileNotFoundError("Provide EdNet-KT1.zip and the Contents questions.csv in the paths above.")
    source_hashes = {RAW_DATA.name: file_sha256(RAW_DATA), QUESTIONS_FILE.name: file_sha256(QUESTIONS_FILE)}
    cached = load_cache(source_hashes)
    if cached is not None:
        return cached
    questions = pd.read_csv(QUESTIONS_FILE)
    if questions["question_id"].duplicated().any():
        raise ValueError("Duplicate question IDs in the answer key.")
    questions = questions.sort_values("question_id", key=lambda values: values.str[1:].astype(int)).reset_index(drop=True)
    item_lookup = dict(zip(questions["question_id"], questions.index))
    answer_lookup = dict(zip(questions["question_id"], questions["correct_answer"]))
    items = questions.rename(columns={"question_id": "original_id"}).copy()
    items.insert(0, "item", np.arange(len(items)))
    # Metadata below is saved for audit and later analyses; the models receive none of it.
    parts, rows_read, rows_kept, invalid_answers = [], 0, 0, 0
    with zipfile.ZipFile(RAW_DATA) as archive:
        members = sorted([name for name in archive.namelist() if Path(name).name.startswith("u") and name.endswith(".csv")],
                         key=lambda name: int(Path(name).stem[1:]))
        if RUN_MODE == "smoke":
            members = members[:300]
        people = pd.DataFrame({"person": np.arange(len(members)), "original_id": [Path(name).stem for name in members]})
        for chunk_start in range(0, len(members), 2000):
            part = CACHE_DIR / f"part_{chunk_start // 2000:05d}.npy"
            marker = part.with_suffix(".json")
            if part.exists() and marker.exists():
                info = json.loads(marker.read_text())
                if info["source_hashes"] != source_hashes or file_sha256(part) != info["sha256"]:
                    raise RuntimeError("A preprocessing checkpoint does not match its input.")
            else:
                blocks, raw_count, invalid_count = [], 0, 0
                for person in range(chunk_start, min(chunk_start + 2000, len(members))):
                    with archive.open(members[person]) as stream:
                        frame = pd.read_csv(stream, usecols=["timestamp", "question_id", "user_answer"])
                    raw_count += len(frame)
                    if not frame["question_id"].isin(item_lookup).all():
                        raise ValueError(f"An unknown question ID occurs in {members[person]}")
                    # Stable order resolves equal timestamps by the last row in the source file.
                    frame = frame.sort_values("timestamp", kind="stable").drop_duplicates("question_id", keep="last")
                    valid = frame["user_answer"].isin(["a", "b", "c", "d"])
                    invalid_count += int((~valid).sum())
                    frame = frame.loc[valid].copy()
                    frame["item"] = frame["question_id"].map(item_lookup)
                    frame = frame.sort_values("item")
                    block = np.empty(len(frame), dtype=RESPONSE_DTYPE)
                    block["person"] = person
                    block["item"] = frame["item"].to_numpy()
                    block["correct"] = (frame["user_answer"] == frame["question_id"].map(answer_lookup)).to_numpy()
                    block["timestamp"] = frame["timestamp"].to_numpy()
                    blocks.append(block)
                block = np.concatenate(blocks)
                temporary = part.with_name(part.stem + ".pending.npy")
                np.save(temporary, block)
                temporary.replace(part)
                info = dict(source_hashes=source_hashes, rows_read=raw_count, rows_kept=len(block),
                            invalid_answers=invalid_count, sha256=file_sha256(part))
                save_json(marker, info)
            rows_read += info["rows_read"]
            rows_kept += info["rows_kept"]
            invalid_answers += info["invalid_answers"]
            parts.append(part)
            print(f"Students {min(chunk_start + 2000, len(members)):,}/{len(members):,}; retained {rows_kept:,} responses", flush=True)
    return finish_cache(parts, people, items, {
        "dataset": DATASET_NAME, "source_hashes": source_hashes,
        "deduplication": "Last response per person–item pair (stable timestamp order).",
        "timestamp_available": True, "original_rows": rows_read, "invalid_answers_excluded": invalid_answers,
    })

responses, data_description = prepare_data()
display(pd.DataFrame([data_description]).T.rename(columns={0: "value"}))

def build_fixed_item_metadata():
    items = pd.read_csv(CACHE_DIR / "items.csv").sort_values("item").reset_index(drop=True)
    np.testing.assert_array_equal(items["item"].to_numpy(), np.arange(data_description["items"]))
    parts = items["part"].astype(int).to_numpy()
    if not np.isin(parts, np.arange(1, 8)).all():
        raise ValueError("EdNet part must be in 1..7")
    part_features = np.eye(7, dtype=np.float32)[parts - 1]
    tag_sets = []
    vocabulary = set()
    for value in items["tags"].fillna("").astype(str):
        tags = tuple(sorted({tag.strip() for tag in value.split(";") if tag.strip()}, key=lambda x: int(x)))
        tag_sets.append(tags)
        vocabulary.update(tags)
    vocabulary = sorted(vocabulary, key=lambda x: int(x))
    tag_index = {tag: index for index, tag in enumerate(vocabulary)}
    tag_features = np.zeros((len(items), len(vocabulary)), dtype=np.float32)
    for row, tags in enumerate(tag_sets):
        for tag in tags:
            tag_features[row, tag_index[tag]] = 1.0
    features = np.concatenate([part_features, tag_features], axis=1)
    digest = hashlib.sha256(features.tobytes(order="C")).hexdigest()
    audit_dir = WORK_DIR / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    save_json(audit_dir / "item_feature_manifest.json", {
        "rows": int(features.shape[0]), "columns": int(features.shape[1]),
        "part_columns": [f"part_{j}" for j in range(1, 8)],
        "tag_vocabulary": vocabulary, "matrix_sha256": digest,
        "excluded_metadata": ["bundle_id", "explanation_id", "deployed_at"],
        "response_derived_features": False,
    })
    return features, digest

ITEM_FEATURES, ITEM_FEATURE_SHA256 = build_fixed_item_metadata()
print("Fixed item metadata:", ITEM_FEATURES.shape, ITEM_FEATURE_SHA256, flush=True)



# CELL 18
def make_split(data, description, config, seed):
    n_people, n_items = description["people"], description["items"]
    people, items = data["person"], data["item"]
    counts = np.bincount(people, minlength=n_people)
    offsets = np.concatenate(([0], np.cumsum(counts)))
    required = max(config["support_counts"]) + config["query_count"]
    eligible = np.flatnonzero(counts >= required)
    if len(eligible) < config["evaluation_students"]:
        raise ValueError(f"Need {config['evaluation_students']} people with ≥{required} distinct responses; found {len(eligible)}.")
    rng = np.random.default_rng(seed)
    evaluation_people = np.sort(rng.choice(eligible, config["evaluation_students"], replace=False))
    is_evaluation_person = np.zeros(n_people, dtype=bool)
    is_evaluation_person[evaluation_people] = True
    # Split fitting/validation rows once. Nothing is selected using test performance.
    validation = rng.random(len(data), dtype=np.float32) < config["validation_fraction"]
    cold_available = ~is_evaluation_person[people]
    cold_fit = cold_available & ~validation
    successes = np.bincount(items[cold_fit], weights=data["correct"][cold_fit], minlength=n_items)
    trials = np.bincount(items[cold_fit], minlength=n_items)
    usable_item = (successes > 0) & (successes < trials)
    # Constant items have no finite unpenalized difficulty estimate. This train-only
    # exclusion applies to every model and to both evaluation protocols.
    usable_rows = usable_item[items]
    support, query = [], []
    for person in evaluation_people:
        rows = np.arange(offsets[person], offsets[person + 1])
        rows = rows[usable_rows[rows]]
        if len(rows) < required:
            raise ValueError(f"Person {person} has only {len(rows)} responses to estimable items; need {required}. No automatic cohort change.")
        chosen = rng.choice(rows, required, replace=False)
        support.append(chosen[:max(config["support_counts"])])
        query.append(chosen[max(config["support_counts"]):])
    support, query = np.array(support), np.array(query)
    warm_available = usable_rows.copy()
    warm_available[query.ravel()] = False
    cold_available &= usable_rows
    plans = {}
    for protocol, available in [("warm", warm_available), ("cold", cold_available)]:
        train = np.flatnonzero(available & ~validation)
        valid = np.flatnonzero(available & validation)
        assert not np.isin(query.ravel(), train).any()
        assert not np.isin(query.ravel(), valid).any()
        if protocol == "cold":
            assert not is_evaluation_person[people[train]].any()
            assert not is_evaluation_person[people[valid]].any()
        assert not np.intersect1d(support.ravel(), query.ravel()).size
        plans[protocol] = {"train": train, "valid": valid}
    return {"people": evaluation_people, "support": support, "query": query,
            "plans": plans, "usable_item": usable_item,
            "audit": {"eligible_people_before_item_filter": len(eligible),
                      "evaluation_people": len(evaluation_people),
                      "query_per_person": config["query_count"],
                      "items_with_no_training_response": int((trials == 0).sum()),
                      "items_with_only_one_training_outcome": int(((trials > 0) & ~usable_item).sum()),
                      "excluded_rows": int((~usable_rows).sum()),
                      "total_rows": len(data)}}


# CELL 21
def irt_grid(nodes):
    theta = np.linspace(-4.0, 4.0, nodes)
    prior = np.exp(-0.5 * theta ** 2)
    return theta, prior / prior.sum()


# CELL 23
def sparse_responses(data, rows, n_people, n_items):
    people = data["person"][rows]
    items = data["item"][rows]
    observed = sparse.csr_matrix((np.ones(len(rows)), (people, items)), shape=(n_people, n_items))
    correct = sparse.csr_matrix((data["correct"][rows].astype(float), (people, items)), shape=(n_people, n_items))
    if observed.data.max() != 1:
        raise ValueError("Repeated person–item pairs must be resolved before fitting.")
    return observed, correct


# CELL 25
def irt_expectation(observed, correct, a, b, theta, prior, person_batch):
    probability = np.clip(expit(a[:, None] * (theta[None, :] - b[:, None])), 1e-9, 1 - 1e-9)
    log_odds = np.log(probability) - np.log1p(-probability)
    log_incorrect = np.log1p(-probability)
    expected_trials = np.zeros_like(probability)
    expected_correct = np.zeros_like(probability)
    likelihood, last_message = 0.0, time.monotonic()
    for start in range(0, observed.shape[0], person_batch):
        obs = observed[start:start + person_batch]
        yes = correct[start:start + person_batch]
        log_posterior = yes @ log_odds + obs @ log_incorrect + np.log(prior)[None, :]
        normalizer = logsumexp(log_posterior, axis=1)
        posterior = np.exp(log_posterior - normalizer[:, None])
        likelihood += float(normalizer.sum())
        expected_trials += obs.T @ posterior
        expected_correct += yes.T @ posterior
        if time.monotonic() - last_message > 30:
            print(f"  IRT expectation: {min(start + person_batch, observed.shape[0]):,}/{observed.shape[0]:,} people", flush=True)
            last_message = time.monotonic()
    return expected_trials, expected_correct, likelihood


# CELL 27
def fit_irt(data, rows, description, config, model, checkpoint):
    observed, correct = sparse_responses(data, rows, description["people"], description["items"])
    theta, prior = irt_grid(config["irt_nodes"])
    trials = np.asarray(observed.sum(axis=0)).ravel()
    successes = np.asarray(correct.sum(axis=0)).ravel()
    active = trials > 0
    initial_rate = np.clip(np.divide(successes, trials, out=np.full_like(trials, 0.5), where=active), .01, .99)
    a, b = np.ones(len(trials)), -np.log(initial_rate / (1 - initial_rate))
    first_iteration, history = 0, []
    checkpoint = Path(checkpoint)
    if RESUME and checkpoint.exists():
        saved = np.load(checkpoint)
        a, b = saved["a"], saved["b"]
        first_iteration = int(saved["iteration"])
        history = saved["history"].tolist()
        print(f"Resuming {model} at iteration {first_iteration}")
    converged = bool(history and history[-1][1] < config["irt_tolerance"])
    iteration = first_iteration - 1
    for iteration in range(first_iteration, config["irt_max_iterations"]):
        if converged:
            iteration = first_iteration - 1
            break
        expected_trials, expected_correct, likelihood = irt_expectation(
            observed, correct, a, b, theta, prior, config["irt_person_batch"])
        old_a, old_b = a.copy(), b.copy()
        for step in range(25):
            probability = np.clip(expit(a[:, None] * (theta[None, :] - b[:, None])), 1e-9, 1 - 1e-9)
            weight = expected_trials * probability * (1 - probability)
            residual = expected_correct - expected_trials * probability
            gradient_b = (-a[:, None] * residual).sum(axis=1)
            hessian_b = -(a[:, None] ** 2 * weight).sum(axis=1)
            b -= gradient_b / np.minimum(hessian_b, -1e-9)
            if model == "2PL":
                probability = np.clip(expit(a[:, None] * (theta[None, :] - b[:, None])), 1e-9, 1 - 1e-9)
                weight = expected_trials * probability * (1 - probability)
                residual = expected_correct - expected_trials * probability
                distance = theta[None, :] - b[:, None]
                gradient_a = (residual * distance).sum(axis=1)
                hessian_a = -(weight * distance ** 2).sum(axis=1)
                a = np.clip(a - gradient_a / np.minimum(hessian_a, -1e-9), .1, 4.0)
        change = float(max(np.max(abs(a - old_a)), np.max(abs(b - old_b))))
        if not np.isfinite(change):
            raise FloatingPointError(f"Nonfinite {model} update")
        history.append([iteration + 1, change, likelihood])
        converged = change < config["irt_tolerance"]
        if iteration == 0 or (iteration + 1) % 10 == 0 or converged:
            print(f"{model}: iteration {iteration+1}, max parameter change={change:.3g}", flush=True)
            save_npz(checkpoint, a=a, b=b, iteration=iteration + 1, history=np.asarray(history))
        if converged:
            break
    del observed, correct
    gc.collect()
    return {"a": a, "b": b, "active": active, "converged": converged,
            "iterations": iteration + 1, "history": np.asarray(history),
            "slope_boundary_items": int((((a <= .1) | (a >= 4.0)) & active).sum())}


# CELL 29
def irt_predict(data, support_rows, query_rows, fitted, config):
    theta, prior = irt_grid(config["irt_nodes"])
    item_probability = np.clip(expit(fitted["a"][:, None] * (theta[None, :] - fitted["b"][:, None])), 1e-9, 1 - 1e-9)
    predictions, estimates = [], []
    for support, query in zip(support_rows, query_rows):
        probability = item_probability[data["item"][support]]
        labels = data["correct"][support, None]
        log_posterior = (labels * np.log(probability) + (1 - labels) * np.log1p(-probability)).sum(axis=0) + np.log(prior)
        posterior = np.exp(log_posterior - logsumexp(log_posterior))
        predictions.append(item_probability[data["item"][query]] @ posterior)
        estimates.append(posterior @ theta)
    return np.asarray(predictions), np.asarray(estimates)


# CELL 32
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, concatenate, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.losses import Loss
from keras import layers, utils

tf.get_logger().setLevel("ERROR")
for physical_gpu in tf.config.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(physical_gpu, True)
print("TensorFlow", tf.__version__, "| GPUs:", len(tf.config.list_physical_devices("GPU")))


# CELL 34
class ValueOfIndexLayer(tf.keras.layers.Layer):
    def __init__(self, index):
        super(ValueOfIndexLayer, self).__init__()
        self.index = index
        self.value = tf.Variable(initial_value=tf.zeros((1,)))

    def call(self, input):
        self.value = input[:, self.index]
        return self.value


# CELL 36
class IndexedValueLayer(layers.Layer):
    def __init__(self, value_shape, trainable: bool, name: str,
                 latent_regularizer=None, latent_constraint=None,
                 latent_initializer="he_normal", **kwargs):
        super(IndexedValueLayer, self).__init__(name=name, **kwargs)
        vocab_size, latent_dims = value_shape
        self.input_dim = vocab_size
        self.output_dim = latent_dims
        self.kernel = self.add_weight(
            shape=(self.input_dim, self.output_dim),
            name=f"{self.name}.kernel", trainable=trainable,
            regularizer=latent_regularizer, constraint=latent_constraint,
            initializer=latent_initializer)

    def call(self, inputs):
        return tf.gather(self.kernel, tf.cast(inputs, tf.int32))

    def set_value(self, value):
        self.kernel.assign(value)

    def get_value(self):
        return self.kernel.numpy()


# CELL 38
class BatchBinaryCrossentropy(Loss):
    def call(self, y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        # label-shape fix carried in the published repo (GY, 2026-07-28)
        y_true = tf.reshape(tf.cast(y_true, y_pred.dtype), tf.shape(y_pred))
        bce = -(y_true * tf.math.log(y_pred) + (1 - y_true) * tf.math.log(1 - y_pred))
        return tf.reduce_mean(bce, axis=-1)


# CELL 40
def build_finder_model(user_latents_shape, item_features_shape, item_latents_shape,
                       user_latent_trainable=True, item_latent_trainable=True):
    inp = Input((2,), name="input")
    user_index = ValueOfIndexLayer(0)(inp)
    item_index = ValueOfIndexLayer(1)(inp)

    user_latents_layer = IndexedValueLayer(user_latents_shape, trainable=user_latent_trainable, name="user_latents")
    item_features_layer = IndexedValueLayer(item_features_shape, trainable=False, name="item_features")
    item_latents_layer = IndexedValueLayer(item_latents_shape, trainable=item_latent_trainable, name="item_latents")

    user_latent_x = user_latents_layer(user_index)
    item_feature_x = item_features_layer(item_index)
    item_latent_x = item_latents_layer(item_index)

    user_latent_x = Dense(96, activation="swish", use_bias=True, kernel_regularizer=l2(0.0001), name="user_latent")(user_latent_x)
    user_latent_x = BatchNormalization()(user_latent_x)
    user_latent_x = Dropout(0.2)(user_latent_x)

    item_feature_x = Dense(64, activation="swish", use_bias=True, kernel_regularizer=l2(0.0001), name="item_feature")(item_feature_x)
    item_feature_x = BatchNormalization()(item_feature_x)
    item_feature_x = Dropout(0.2)(item_feature_x)

    item_latent_x = Dense(96, activation="swish", use_bias=True, kernel_regularizer=l2(0.0001), name="item_latent")(item_latent_x)
    item_latent_x = BatchNormalization()(item_latent_x)
    item_latent_x = Dropout(0.2)(item_latent_x)

    x = concatenate([user_latent_x, item_feature_x, item_latent_x])
    x = BatchNormalization()(x)
    x = Dense(256, activation="swish", use_bias=True, kernel_regularizer=l2(0.0001))(x)
    x = BatchNormalization()(x); x = Dropout(0.3)(x)
    x = Dense(128, activation="swish", use_bias=True, kernel_regularizer=l2(0.0001))(x)
    x = BatchNormalization()(x); x = Dropout(0.3)(x)
    x = Dense(64, activation="swish", use_bias=True, kernel_regularizer=l2(0.0001))(x)
    x = BatchNormalization()(x); x = Dropout(0.2)(x)
    score_output = Dense(1, activation="sigmoid", kernel_regularizer=l2(0.0001))(x)

    model = Model(inp, score_output)
    model.user_latents_layer = user_latents_layer
    model.item_features_layer = item_features_layer
    model.item_latents_layer = item_latents_layer
    return model


# CELL 42
def build_predictor_model_optimized(user_feature_dim, item_feature_dim, latent_dim, name="Predictor"):
    user_latent_input = Input((latent_dim,), name="user_latent_input")
    item_feature_input = Input((item_feature_dim,), name="item_feature_input")
    item_latent_input = Input((latent_dim,), name="item_latent_input")
    user_feature_input = Input((user_feature_dim,), name="user_feature_input")

    user_latent_x = Dense(96, activation="swish", use_bias=True, kernel_regularizer=l2(0.0001), name="user_latent")(user_latent_input)
    user_latent_x = BatchNormalization()(user_latent_x); user_latent_x = Dropout(0.2)(user_latent_x)

    item_feature_x = Dense(64, activation="swish", use_bias=True, kernel_regularizer=l2(0.0001), name="item_feature")(item_feature_input)
    item_feature_x = BatchNormalization()(item_feature_x); item_feature_x = Dropout(0.2)(item_feature_x)

    item_latent_x = Dense(96, activation="swish", use_bias=True, kernel_regularizer=l2(0.0001), name="item_latent")(item_latent_input)
    item_latent_x = BatchNormalization()(item_latent_x); item_latent_x = Dropout(0.2)(item_latent_x)

    user_feature_x = Dense(4, activation="swish", use_bias=True, kernel_regularizer=l2(0.0001), name="user_feature")(user_feature_input)
    user_feature_x = BatchNormalization()(user_feature_x); user_feature_x = Dropout(0.1)(user_feature_x)

    x = concatenate([user_latent_x, item_feature_x, item_latent_x, user_feature_x])
    x = BatchNormalization()(x)
    x = Dense(256, activation="swish", use_bias=True, kernel_regularizer=l2(0.0001))(x)
    x = BatchNormalization()(x); x = Dropout(0.3)(x)
    x = Dense(128, activation="swish", use_bias=True, kernel_regularizer=l2(0.0001))(x)
    x = BatchNormalization()(x); x = Dropout(0.3)(x)
    x = Dense(64, activation="swish", use_bias=True, kernel_regularizer=l2(0.0001))(x)
    x = BatchNormalization()(x); x = Dropout(0.2)(x)
    score_output = Dense(1, activation="sigmoid", kernel_regularizer=l2(0.0001))(x)

    return Model(inputs=[user_latent_input, item_feature_input, item_latent_input, user_feature_input],
                 outputs=score_output, name=name)


# CELL 44
class ResponseBatches(tf.keras.utils.PyDataset):
    """Read only one batch of index pairs from the memory-mapped response log."""
    def __init__(self, data, rows, batch_size, shuffle, seed):
        super().__init__()
        self.data, self.rows = data, np.asarray(rows).copy()
        self.batch_size, self.shuffle = batch_size, shuffle
        self.rng = np.random.default_rng(seed)
        self.on_epoch_end()
    def __len__(self):
        return (len(self.rows) + self.batch_size - 1) // self.batch_size
    def __getitem__(self, index):
        rows = self.rows[index * self.batch_size:(index + 1) * self.batch_size]
        inputs = np.column_stack((self.data["person"][rows], self.data["item"][rows])).astype(np.float32)
        return inputs, self.data["correct"][rows].astype(np.float32).reshape(-1, 1)
    def on_epoch_end(self):
        if self.shuffle:
            self.rng.shuffle(self.rows)


# CELL 46
class TrainingProgress(tf.keras.callbacks.Callback):
    def __init__(self, label, interval=30):
        super().__init__()
        self.label, self.interval = label, interval
    def on_train_begin(self, logs=None):
        self.last_message = time.monotonic()
        print(f"Starting {self.label}", flush=True)
    def on_train_batch_end(self, batch, logs=None):
        if time.monotonic() - self.last_message >= self.interval:
            print(f"  {self.label}: batch {batch+1}, loss={logs['loss']:.5f}", flush=True)
            self.last_message = time.monotonic()
    def on_epoch_end(self, epoch, logs=None):
        if epoch == 0 or (epoch + 1) % 10 == 0 or "val_auc" in logs:
            values = {name:round(float(value), 5) for name,value in logs.items()}
            print(f"  {self.label}: epoch {epoch+1} {values}", flush=True)


# CELL 48
class BCEPredictor(tf.keras.Model):
    """Use the reference Predictor objective: BCE, without declared L2 terms."""
    def __init__(self, inputs, outputs):
        super().__init__(inputs=inputs, outputs=outputs)
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.accuracy_tracker = tf.keras.metrics.BinaryAccuracy(name="accuracy")
        self.auc_tracker = tf.keras.metrics.AUC(name="auc")
    @property
    def metrics(self):
        return [self.loss_tracker, self.accuracy_tracker, self.auc_tracker]
    def train_step(self, batch):
        inputs, labels, weights = tf.keras.utils.unpack_x_y_sample_weight(batch)
        with tf.GradientTape() as tape:
            probabilities = self(inputs, training=True)
            loss_value = self.loss(labels, probabilities)
        gradients = tape.gradient(loss_value, self.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_weights))
        return self.update_metrics(labels, probabilities, loss_value)
    def test_step(self, batch):
        inputs, labels, weights = tf.keras.utils.unpack_x_y_sample_weight(batch)
        probabilities = self(inputs, training=False)
        return self.update_metrics(labels, probabilities, self.loss(labels, probabilities))
    def update_metrics(self, labels, probabilities, loss_value):
        self.loss_tracker.update_state(loss_value)
        self.accuracy_tracker.update_state(labels, probabilities)
        self.auc_tracker.update_state(labels, probabilities)
        return {metric.name:metric.result() for metric in self.metrics}


# CELL 50
class ReferenceStopping(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.best, self.best_weights = 0.0, None
        self.stop_count = self.rate_count = 0
    def on_epoch_end(self, epoch, logs=None):
        if logs["val_auc"] > self.best:
            self.best, self.best_weights = logs["val_auc"], self.model.get_weights()
            self.stop_count = self.rate_count = 0
        else:
            self.stop_count += 1
            self.rate_count += 1
            if self.rate_count >= 3:
                reduced = float(self.model.optimizer.learning_rate.numpy()) * .5
                if reduced > 1e-6:
                    self.model.optimizer.learning_rate.assign(reduced)
                    self.rate_count = 0
        if self.stop_count >= 7:
            self.model.stop_training = True
    def on_train_end(self, logs=None):
        if self.best_weights is not None:
            self.model.set_weights(self.best_weights)


# CELL 52
def make_indexed_predictor(user_latents, item_latents, item_features, config):
    inputs = Input((2,), name="response_indices")
    user_index, item_index = ValueOfIndexLayer(0)(inputs), ValueOfIndexLayer(1)(inputs)
    user_layer = IndexedValueLayer(user_latents.shape, trainable=False, name="fixed_user_latents")
    item_layer = IndexedValueLayer(item_latents.shape, trainable=False, name="fixed_item_latents")
    feature_layer = IndexedValueLayer(item_features.shape, trainable=False, name="fixed_item_features")
    user_zeros = layers.Lambda(lambda values: tf.zeros_like(values[:, :1]), name="zero_user_features")(inputs)
    core = build_predictor_model_optimized(1, item_features.shape[1], config["latent_size"])
    output = core([user_layer(user_index), feature_layer(item_index), item_layer(item_index), user_zeros])
    model = BCEPredictor(inputs, output)
    user_layer.set_value(user_latents)
    item_layer.set_value(item_latents)
    feature_layer.set_value(item_features)
    model.user_latents_layer, model.item_latents_layer = user_layer, item_layer
    model.item_features_layer = feature_layer
    return model


# CELL 54


# Observational checkpointing; no training update or stopping rule is changed.
CHECKPOINT_DIRECTORY = None


# CELL 56
def save_training_state(model, label, include_optimizer=True):
    if CHECKPOINT_DIRECTORY is None:
        return
    values = {f"model_{j}":v.numpy() for j,v in enumerate(model.variables)}
    metadata = {"model":[{"name":v.path,"shape":list(v.shape)} for v in model.variables]}
    if include_optimizer:
        values.update({f"optimizer_{j}":v.numpy() for j,v in enumerate(model.optimizer.variables)})
        metadata["optimizer"] = [{"name":v.path,"shape":list(v.shape)} for v in model.optimizer.variables]
    else:
        metadata["optimizer_note"] = "Predictor best weights restored by original callback; end optimizer is not paired with them and is not exported."
    path = CHECKPOINT_DIRECTORY / (label + "_state.npz")
    temporary = path.with_suffix(".pending.npz")
    np.savez(temporary, **values)
    temporary.replace(path)
    save_json(CHECKPOINT_DIRECTORY / (label + "_state.json"), metadata)


# CELL 58

OriginalTrainingProgress = TrainingProgress


# CELL 60
class TrainingProgress(OriginalTrainingProgress):
    def on_train_begin(self, logs=None):
        super().on_train_begin(logs)
        self.phase_started = time.monotonic()
    def on_train_end(self, logs=None):
        elapsed = time.monotonic() - self.phase_started
        if CHECKPOINT_DIRECTORY is None:
            return
        start = time.monotonic()
        if self.label.startswith("cycle"):
            save_training_state(self.model, "last_" + self.label.split()[-1])
        elif self.label == "Predictor":
            save_training_state(self.model, "best_Predictor", include_optimizer=False)
        path = CHECKPOINT_DIRECTORY / "phase_times.json"
        phases = json.loads(path.read_text()) if path.exists() else []
        phases.append({"phase":self.label,"fit_seconds":elapsed,"checkpoint_seconds":time.monotonic()-start})
        save_json(path, phases)


# CELL 62
def fit_cdld(data, rows, valid_rows, description, config, seed):
    tf.keras.utils.set_random_seed(seed)
    n_people, n_items = description["people"], description["items"]
    size = config["latent_size"]
    initial_users = np.random.default_rng(seed).random((n_people, size)).astype(np.float32)
    initial_items = np.random.default_rng(seed + 1000).random((n_items, size)).astype(np.float32)
    item_features = ITEM_FEATURES
    if item_features.shape[0] != n_items:
        raise ValueError("Item metadata row count mismatch")
    user_finder = build_finder_model(initial_users.shape, item_features.shape, initial_items.shape, True, False)
    item_finder = build_finder_model(initial_users.shape, item_features.shape, initial_items.shape, False, True)
    for finder in [user_finder, item_finder]:
        finder.user_latents_layer.set_value(initial_users)
        finder.item_features_layer.set_value(item_features)
        finder.item_latents_layer.set_value(initial_items)
        finder.compile(optimizer=tf.keras.optimizers.Adam(config["learning_rate"]), loss=BatchBinaryCrossentropy(),
                       metrics=[tf.keras.metrics.BinaryAccuracy(name="accuracy"), tf.keras.metrics.AUC(name="auc")])
    training = ResponseBatches(data, rows, config["batch_size"], True, seed)
    validation = ResponseBatches(data, valid_rows, config["batch_size"], False, seed)
    schedule = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=.5, patience=2, min_lr=1e-5)
    histories = []
    for cycle in range(config["finder_cycles"]):
        user_finder.item_latents_layer.set_value(item_finder.item_latents_layer.get_value())
        for label, finder in [("ULD", user_finder), ("ILD", item_finder)]:
            if label == "ILD":
                item_finder.user_latents_layer.set_value(user_finder.user_latents_layer.get_value())
            history = finder.fit(training, validation_data=validation, epochs=config["finder_subepochs"], verbose=0,
                                 callbacks=[schedule, TrainingProgress(f"cycle {cycle+1} {label}")])
            histories.append({"phase":label, "cycle":cycle+1, "values":history.history})
    final_users = user_finder.user_latents_layer.get_value()
    final_items = item_finder.item_latents_layer.get_value()
    user_finder.item_latents_layer.set_value(final_items)
    del item_finder
    gc.collect()
    predictor = make_indexed_predictor(final_users, final_items, item_features, config)
    predictor.compile(optimizer=tf.keras.optimizers.Adam(config["learning_rate"]), loss=BatchBinaryCrossentropy())
    history = predictor.fit(training, validation_data=validation, epochs=config["predictor_epochs"], verbose=0,
                            callbacks=[ReferenceStopping(), TrainingProgress("Predictor")])
    histories.append({"phase":"Predictor", "cycle":0, "values":history.history})
    return {"finder":user_finder, "predictor":predictor, "initial_users":initial_users,
            "final_users":final_users, "final_items":final_items, "histories":histories}


# CELL 64
def predict_cdld(fitted, data, query_rows, config):
    flat = query_rows.ravel()
    inputs = np.column_stack((data["person"][flat], data["item"][flat])).astype(np.float32)
    return fitted["predictor"].predict(inputs, batch_size=config["batch_size"], verbose=0).reshape(query_rows.shape)


# CELL 66
def adapt_cdld(fitted, data, people, support_rows, config):
    finder = fitted["finder"]
    for layer in finder.layers:
        layer.trainable = False
    finder.user_latents_layer.trainable = True
    finder.compile(optimizer=tf.keras.optimizers.Adam(config["learning_rate"]), loss=BatchBinaryCrossentropy())
    if len(finder.trainable_weights) != 1:
        raise RuntimeError("Only the user-latent array may be adapted.")
    before = [(weight, weight.numpy().copy()) for layer in finder.layers
              if layer is not finder.user_latents_layer for weight in layer.weights]
    values = fitted["final_users"].copy()
    values[people] = fitted["initial_users"][people]
    finder.user_latents_layer.set_value(values)
    if support_rows.shape[1]:
        flat = support_rows.ravel()
        inputs = np.column_stack((data["person"][flat], data["item"][flat])).astype(np.float32)
        labels = data["correct"][flat].astype(np.float32).reshape(-1, 1)
        finder.fit(inputs, labels, batch_size=len(flat), epochs=config["adaptation_epochs"], verbose=0,
                   callbacks=[TrainingProgress(f"adaptation k={support_rows.shape[1]}")])
    for weight, old_value in before:
        np.testing.assert_array_equal(weight.numpy(), old_value)
    adapted = finder.user_latents_layer.get_value()
    other_people = np.ones(len(values), dtype=bool)
    other_people[people] = False
    np.testing.assert_array_equal(adapted[other_people], values[other_people])
    fitted["predictor"].user_latents_layer.set_value(adapted)
    return adapted[people]


# CELL 69
def run_reproduction_gates():
    # Sparse likelihood: compare with a literal per-response calculation,
    # including missing entries and a person with no observed response.
    small = np.empty(4, dtype=RESPONSE_DTYPE)
    small["person"], small["item"], small["correct"] = [0, 0, 1, 1], [0, 2, 1, 2], [1, 0, 0, 1]
    small["timestamp"] = -1
    observed, correct = sparse_responses(small, np.arange(4), 3, 3)
    theta, prior = irt_grid(41)
    a, b = np.array([.7, 1.1, 1.5]), np.array([-.3, .2, 1.0])
    trials, successes, likelihood = irt_expectation(observed, correct, a, b, theta, prior, 2)
    reference_trials, reference_successes = np.zeros_like(trials), np.zeros_like(successes)
    for person in range(3):
        rows = np.flatnonzero(small["person"] == person)
        probability = expit(a[small["item"][rows], None] * (theta[None, :] - b[small["item"][rows], None]))
        labels = small["correct"][rows, None]
        log_probability = (labels * np.log(probability) + (1-labels) * np.log1p(-probability)).sum(axis=0) + np.log(prior)
        posterior = np.exp(log_probability - logsumexp(log_probability))
        for row in rows:
            reference_trials[small["item"][row]] += posterior
            reference_successes[small["item"][row]] += small["correct"][row] * posterior
    np.testing.assert_allclose(trials, reference_trials, atol=1e-12)
    np.testing.assert_allclose(successes, reference_successes, atol=1e-12)
    # BCE shape and an explicit BCE-only gradient comparison.
    loss = BatchBinaryCrossentropy()
    np.testing.assert_allclose(float(loss(tf.constant([0., 1.]), tf.constant([[.2], [.8]]))), -np.log(.8), rtol=1e-6)
    tf.keras.utils.set_random_seed(91)
    inputs = Input((2,))
    output = Dense(1, activation="sigmoid", kernel_regularizer=l2(.1))(inputs)
    model = BCEPredictor(inputs, output)
    model.compile(optimizer=tf.keras.optimizers.SGD(.01), loss=loss)
    values, labels = tf.constant([[1., 2.], [2., 1.]]), tf.constant([[0.], [1.]])
    original = [weight.numpy().copy() for weight in model.trainable_weights]
    with tf.GradientTape() as tape:
        expected_loss = loss(labels, model(values, training=True))
    gradients = tape.gradient(expected_loss, model.trainable_weights)
    model.train_step((values, labels))
    for old, gradient, weight in zip(original, gradients, model.trainable_weights):
        np.testing.assert_allclose(weight.numpy(), old - .01 * gradient.numpy(), atol=1e-7)
    print("PASS: missing-response likelihood, label shape, and BCE-only Predictor update.")
    tf.keras.backend.clear_session()
    gc.collect()
    return True

GATES_PASSED = run_reproduction_gates()


# CELL 72
def response_metrics(actual, probability):
    labels, predicted = actual.ravel(), np.clip(probability.ravel(), 1e-7, 1 - 1e-7)
    observed_score, predicted_score = actual.mean(axis=1), probability.mean(axis=1)
    correlation = float(np.corrcoef(observed_score, predicted_score)[0, 1]) if predicted_score.std() > 1e-12 and observed_score.std() > 1e-12 else np.nan
    return {"auc": float(roc_auc_score(labels, predicted)),
            "log_loss": float(log_loss(labels, predicted, labels=[0, 1])),
            "brier": float(brier_score_loss(labels, predicted)),
            "accuracy": float(((predicted >= .5) == labels).mean()),
            "score_correlation": correlation,
            "observed_correct_rate": float(labels.mean()),
            "score_mae": float(abs(observed_score - predicted_score).mean())}

def append_prediction(storage, metric_rows, model, k, probability, actual, protocol, seed):
    if not np.isfinite(probability).all() or (probability < 0).any() or (probability > 1).any():
        raise ValueError(f"Invalid predictions: {model}, k={k}")
    storage[f"probability|{model}|{k}"] = probability.astype(np.float32)
    metric_rows.append({"protocol":protocol, "seed":seed, "model":model, "k":k,
                        **response_metrics(actual, probability)})

def verify_completed(directory):
    marker = directory / "complete.json"
    if not marker.exists():
        return False
    checksums = json.loads(marker.read_text())
    for name, expected in checksums.items():
        if file_sha256(directory / name) != expected:
            raise RuntimeError(f"Saved result failed its checksum: {directory / name}")
    return True


# CELL 74
def run_one_experiment(data, description, split, config, seed, protocol, directory):
    directory.mkdir(exist_ok=True, parents=True)
    rows = split["plans"][protocol]["train"]
    valid_rows = split["plans"][protocol]["valid"]
    people, query = split["people"], split["query"]
    actual = data["correct"][query]
    results = {"person_ids":people, "query_items":data["item"][query], "actual":actual,
               "query_row_ids":query, "support_row_ids":split["support"],
               "query_timestamp":data["timestamp"][query]}
    metric_rows, model_diagnostics = [], {}
    save_npz(directory / "split.npz", train_rows=rows, validation_rows=valid_rows,
             evaluation_people=people, query_rows=query, support_rows=split["support"])
    save_json(directory / "split_audit.json", split["audit"])
    if protocol == "warm":
        training_people = data["person"][rows]
        support = [rows[np.searchsorted(training_people, person, side="left"):
                        np.searchsorted(training_people, person, side="right")] for person in people]
        counts = [-1]  # All available training responses; not a k-shot condition.
    else:
        support = split["support"]
        counts = config["support_counts"]
    # Reuse only independently verified, converged IRT fits on identical rows.
    reuse_root = WORK_DIR / "previous_results" / DATASET_NAME / RUN_MODE
    if reuse_root.exists():
        import shutil
        old_config = json.loads((reuse_root / "run_manifest.json").read_text())["config"]
        assert {k:v for k,v in old_config.items() if k != "finder_subepochs"} == {k:v for k,v in config.items() if k != "finder_subepochs"}
        old_manifest = json.loads((reuse_root / "run_manifest.json").read_text())
        assert old_manifest["response_sha256"] == description["response_sha256"]
        old = reuse_root / f"{protocol}_seed{seed}"
        hashes = json.loads((old / "complete.json").read_text())
        for name in ["split.npz", "model_diagnostics.json", "Rasch_checkpoint.npz", "2PL_checkpoint.npz"]:
            assert file_sha256(old / name) == hashes[name], name
        with np.load(old / "split.npz") as before, np.load(directory / "split.npz") as after:
            assert set(before.files) == set(after.files)
            for key in before.files:
                np.testing.assert_array_equal(before[key], after[key])
        diagnostics = json.loads((old / "model_diagnostics.json").read_text())
        for model in ["Rasch", "2PL"]:
            assert diagnostics[model]["converged"]
            with np.load(old / f"{model}_checkpoint.npz") as checkpoint:
                assert checkpoint["history"][-1,1] < config["irt_tolerance"]
            target = directory / f"{model}_checkpoint.npz"
            shutil.copy2(old / target.name, target)
        save_json(directory / "irt_reuse.json", {"source":str(old), "source_manifest_sha256":file_sha256(reuse_root / "run_manifest.json"), "verified_identical_split":True})
    rate = float(data["correct"][rows].mean())
    item_count = np.bincount(data["item"][rows], minlength=description["items"])
    item_correct = np.bincount(data["item"][rows], weights=data["correct"][rows], minlength=description["items"])
    item_rate = np.divide(item_correct, item_count, out=np.full_like(item_correct, rate), where=item_count > 0)
    for k in counts:
        append_prediction(results, metric_rows, "Constant", k, np.full(actual.shape, rate), actual, protocol, seed)
        append_prediction(results, metric_rows, "Item rate", k, item_rate[data["item"][query]], actual, protocol, seed)
    for model in ["Rasch", "2PL"]:
        fitted_irt = fit_irt(data, rows, description, config, model, directory / f"{model}_checkpoint.npz")
        model_diagnostics[model] = {key:fitted_irt[key] for key in ["converged", "iterations", "slope_boundary_items"]}
        pd.DataFrame({"item":np.arange(description["items"]), "a":fitted_irt["a"], "b":fitted_irt["b"],
                      "estimable":fitted_irt["active"]}).to_csv(directory / f"{model}_item_parameters.csv", index=False)
        pd.DataFrame(fitted_irt["history"], columns=["iteration", "max_change", "log_likelihood_before_update"]).to_csv(directory / f"{model}_history.csv", index=False)
        for k in counts:
            chosen = support if protocol == "warm" else support[:, :k]
            probability, theta = irt_predict(data, chosen, query, fitted_irt, config)
            results[f"theta|{model}|{k}"] = theta
            append_prediction(results, metric_rows, model, k, probability, actual, protocol, seed)
        del fitted_irt
        gc.collect()
    global CHECKPOINT_DIRECTORY
    CHECKPOINT_DIRECTORY = directory
    fitted_cdld = fit_cdld(data, rows, valid_rows, description, config, seed)
    save_json(directory / "CDLD_history.json", fitted_cdld["histories"])
    save_training_state(fitted_cdld["finder"], "Finder_before_adaptation", include_optimizer=True)
    np.save(directory / "CDLD_item_latents.npy", fitted_cdld["final_items"])
    # Latents are numeric analysis artifacts; no executable fitted model is exported.
    np.save(directory / "CDLD_user_latents.npy", fitted_cdld["final_users"])
    for k in counts:
        if protocol == "cold":
            results[f"adapted_latents|{k}"] = adapt_cdld(fitted_cdld, data, people, support[:, :k], config)
        probability = predict_cdld(fitted_cdld, data, query, config)
        append_prediction(results, metric_rows, "CDLD", k, probability, actual, protocol, seed)
    save_npz(directory / "predictions.npz", **results)
    pd.DataFrame(metric_rows).to_csv(directory / "metrics.csv", index=False)
    save_json(directory / "model_diagnostics.json", model_diagnostics)
    save_json(directory / "complete.json", {path.name:file_sha256(path) for path in directory.iterdir()
                                          if path.is_file() and path.name != "complete.json"})
    del fitted_cdld
    tf.keras.backend.clear_session()
    gc.collect()


# CELL 76
def run_experiments(data, description, config):
    notebook_document = json.loads(NOTEBOOK_PATH.read_text())
    actual_common_source = "\n".join("".join(cell["source"]) for cell in notebook_document["cells"]
                                     if cell["cell_type"] == "code" and cell.get("metadata", {}).get("shared"))
    actual_common_digest = hashlib.sha256(actual_common_source.encode()).hexdigest()
    manifest = {"dataset":DATASET_NAME, "mode":RUN_MODE, "config":config,
                "response_sha256":description["response_sha256"], "common_code_sha256":actual_common_digest,
                "condition":"item_metadata_part_tags", "item_feature_sha256":ITEM_FEATURE_SHA256}
    manifest_file = OUTPUT_DIR / "run_manifest.json"
    if manifest_file.exists() and json.loads(manifest_file.read_text()) != manifest:
        raise RuntimeError("Different data, code or settings already occupy this output directory. Choose a new WORK_DIR.")
    save_json(manifest_file, manifest)
    failures = []
    for seed in RUN_SEEDS:
        split = make_split(data, description, config, seed)
        for protocol in config["protocols"]:
            directory = OUTPUT_DIR / f"{protocol}_seed{seed}"
            if RESUME and verify_completed(directory):
                print("Verified completed run; skipping:", directory.name)
                continue
            if directory.exists() and not RESUME:
                raise FileExistsError("Existing result directory; enable RESUME or choose a new WORK_DIR.")
            pending = OUTPUT_DIR / (directory.name + ".pending")
            if RESUME and verify_completed(pending):
                pending.rename(directory)
                print("Recovered a completed pending run:", directory.name)
                continue
            try:
                started = time.monotonic()
                run_one_experiment(data, description, split, config, seed, protocol, pending)
                pending.rename(directory)
                print(f"Completed {directory.name} in {(time.monotonic()-started)/60:.1f} minutes", flush=True)
            except Exception as error:
                failures.append({"protocol":protocol, "seed":seed, "error":str(error)})
                save_json(OUTPUT_DIR / f"failure_{protocol}_{seed}.json", {"traceback":traceback.format_exc()})
                print(f"FAILED {protocol}, seed={seed}: {error}. Continuing remaining runs.", flush=True)
            finally:
                tf.keras.backend.clear_session()
                gc.collect()
        del split
        gc.collect()
    if failures:
        raise RuntimeError(f"{len(failures)} run(s) failed. Inspect failure files and rerun this cell to resume.")
    print("All requested runs are complete and checksummed.")


# CELL 78

if RUN_EXPERIMENT:
    assert GATES_PASSED
    run_experiments(responses, data_description, CONFIG)
else:
    print("Training skipped. Continue to the saved-results analysis and figures.")


