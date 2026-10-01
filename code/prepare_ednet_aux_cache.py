"""Build an elapsed-time cache aligned exactly to the verified EdNet response cache."""

import hashlib
import json
from pathlib import Path
import os
WORKSPACE = Path(os.environ.get("CDLD_WORKSPACE", Path(__file__).resolve().parents[1] / "work"))
import time
import zipfile

import numpy as np
import pandas as pd


SOURCE_ROOT = (WORKSPACE / '20260914_cdld_reviewer')
RAW_ZIP = SOURCE_ROOT / "data/EdNet-KT1.zip"
QUESTIONS = SOURCE_ROOT / "data/questions.csv"
REFERENCE_RESPONSES = SOURCE_ROOT / "cache/EdNet/full/responses.npy"
OUTPUT_ROOT = (WORKSPACE / '20260920_representation_study/aux_cache')
CHUNK_PEOPLE = 2000


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".pending")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    temporary.replace(path)


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    reference = np.load(REFERENCE_RESPONSES, mmap_mode="r")
    questions = pd.read_csv(QUESTIONS).sort_values(
        "question_id", key=lambda values: values.str[1:].astype(int)
    ).reset_index(drop=True)
    item_lookup = dict(zip(questions["question_id"], questions.index))
    answer_lookup = dict(zip(questions["question_id"], questions["correct_answer"]))
    source_hashes = {RAW_ZIP.name: sha256(RAW_ZIP), QUESTIONS.name: sha256(QUESTIONS)}
    parts = []
    aligned_rows = 0
    raw_positive = raw_zero = raw_negative = raw_missing = 0
    started = time.time()
    with zipfile.ZipFile(RAW_ZIP) as archive:
        members = sorted(
            [name for name in archive.namelist() if Path(name).name.startswith("u") and name.endswith(".csv")],
            key=lambda name: int(Path(name).stem[1:]),
        )
        for chunk_start in range(0, len(members), CHUNK_PEOPLE):
            part_index = chunk_start // CHUNK_PEOPLE
            output = OUTPUT_ROOT / f"elapsed_part_{part_index:05d}.npy"
            marker = output.with_suffix(".json")
            if output.exists() and marker.exists():
                info = json.loads(marker.read_text())
                if info["source_hashes"] != source_hashes or sha256(output) != info["sha256"]:
                    raise RuntimeError(f"Existing auxiliary part failed validation: {output}")
                aligned_rows += info["rows"]
                raw_positive += info["positive"]
                raw_zero += info["zero"]
                raw_negative += info["negative"]
                raw_missing += info["missing"]
                parts.append(output)
                continue
            elapsed_blocks = []
            reference_start = aligned_rows
            for person in range(chunk_start, min(chunk_start + CHUNK_PEOPLE, len(members))):
                with archive.open(members[person]) as stream:
                    frame = pd.read_csv(
                        stream,
                        usecols=["timestamp", "question_id", "user_answer", "elapsed_time"],
                    )
                frame = frame.sort_values("timestamp", kind="stable").drop_duplicates("question_id", keep="last")
                valid = frame["user_answer"].isin(["a", "b", "c", "d"])
                frame = frame.loc[valid].copy()
                frame["item"] = frame["question_id"].map(item_lookup)
                frame = frame.sort_values("item")
                elapsed = pd.to_numeric(frame["elapsed_time"], errors="coerce").to_numpy(np.float64)
                person_rows = reference[reference_start:reference_start + len(frame)]
                if len(person_rows) != len(frame):
                    raise RuntimeError("Reference cache ended before the rebuilt rows")
                expected_person = np.full(len(frame), person, dtype=np.int32)
                expected_item = frame["item"].to_numpy(np.int32)
                expected_correct = (
                    frame["user_answer"] == frame["question_id"].map(answer_lookup)
                ).to_numpy(np.uint8)
                expected_timestamp = frame["timestamp"].to_numpy(np.int64)
                for field, expected in [
                    ("person", expected_person),
                    ("item", expected_item),
                    ("correct", expected_correct),
                    ("timestamp", expected_timestamp),
                ]:
                    if not np.array_equal(person_rows[field], expected):
                        raise RuntimeError(f"Auxiliary cache alignment failed at person={person}, field={field}")
                reference_start += len(frame)
                raw_missing += int(np.isnan(elapsed).sum())
                raw_positive += int(np.sum(elapsed > 0))
                raw_zero += int(np.sum(elapsed == 0))
                raw_negative += int(np.sum(elapsed < 0))
                elapsed_blocks.append(elapsed)
            elapsed_values = np.concatenate(elapsed_blocks)
            temporary = output.with_name(output.stem + ".pending.npy")
            np.save(temporary, elapsed_values)
            temporary.replace(output)
            info = {
                "source_hashes": source_hashes,
                "people_start": chunk_start,
                "people_stop": min(chunk_start + CHUNK_PEOPLE, len(members)),
                "rows": int(len(elapsed_values)),
                "positive": int(np.sum(elapsed_values > 0)),
                "zero": int(np.sum(elapsed_values == 0)),
                "negative": int(np.sum(elapsed_values < 0)),
                "missing": int(np.isnan(elapsed_values).sum()),
                "sha256": sha256(output),
            }
            write_json(marker, info)
            aligned_rows += len(elapsed_values)
            parts.append(output)
            print(f"people={info['people_stop']}/{len(members)} rows={aligned_rows}", flush=True)
    if aligned_rows != len(reference):
        raise RuntimeError(f"Row count mismatch: auxiliary={aligned_rows}, reference={len(reference)}")
    final_path = OUTPUT_ROOT / "elapsed_time_ms.npy"
    temporary = final_path.with_name(final_path.stem + ".pending.npy")
    output_map = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.float64, shape=(aligned_rows,))
    cursor = 0
    for part in parts:
        values = np.load(part, mmap_mode="r")
        output_map[cursor:cursor + len(values)] = values
        cursor += len(values)
    output_map.flush()
    del output_map
    temporary.replace(final_path)
    values = np.load(final_path, mmap_mode="r")
    finite_positive = np.asarray(values[np.isfinite(values) & (values > 0)])
    quantiles = {
        str(probability): float(np.quantile(finite_positive, probability))
        for probability in [0.0001, 0.001, 0.01, 0.5, 0.99, 0.999, 0.9999]
    }
    completion = {
        "source_hashes": source_hashes,
        "reference_response_sha256": sha256(REFERENCE_RESPONSES),
        "elapsed_time_sha256": sha256(final_path),
        "unit": "milliseconds",
        "rows": int(aligned_rows),
        "positive": int(raw_positive),
        "zero": int(raw_zero),
        "negative": int(raw_negative),
        "missing": int(raw_missing),
        "positive_quantiles_ms": quantiles,
        "elapsed_seconds": time.time() - started,
        "alignment": "Exact equality of person, item, correctness, and timestamp for every retained row.",
    }
    write_json(OUTPUT_ROOT / "complete.json", completion)
    print(json.dumps(completion, indent=2), flush=True)


if __name__ == "__main__":
    main()
