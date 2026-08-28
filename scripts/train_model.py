#!/usr/bin/env python3
"""Train Empwave's multi-label emotion classifier from GoEmotions."""

import argparse
import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_ENCODER = "sentence-transformers/all-MiniLM-L6-v2"
RUNTIME_MIN_CONFIDENCE = 0.93
NEUTRAL_COMPETITION_MARGIN = 0.15
NEUTRAL_OVERRIDE_CONFIDENCE = 0.95
MAX_RUNTIME_EMOTIONS = 5


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Train Empwave's supervised multi-label emotion classifier over "
            "frozen sentence embeddings."
        )
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data") / "processed",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models") / "trained",
    )
    parser.add_argument("--encoder", default=DEFAULT_ENCODER)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-iter", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--validation-limit", type=int)
    parser.add_argument("--test-limit", type=int)
    return parser.parse_args()


def load_split(path, labels, limit=None):
    texts = []
    targets = []
    sample_weights = []

    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if limit is not None and len(texts) >= limit:
                break
            record = json.loads(line)
            source_metadata = record.get("source", {})
            if source_metadata.get("dataset") != "GoEmotions":
                continue
            text = record.get("text", "").strip()
            emotions = record.get(
                "emotions",
                source_metadata.get("emotion_labels"),
            )
            if not text or not isinstance(emotions, list):
                raise ValueError(
                    f"Invalid record at {path}:{line_number}"
                )
            unknown = set(emotions) - set(labels)
            if unknown:
                raise ValueError(
                    f"Unknown emotions at {path}:{line_number}: "
                    f"{sorted(unknown)}"
                )
            texts.append(text)
            targets.append([int(label in emotions) for label in labels])
            sample_weights.append(float(record.get("sample_weight", 1.0)))

    if not texts:
        raise ValueError(f"No training records found in {path}")

    return (
        texts,
        np.asarray(targets, dtype=np.int8),
        np.asarray(sample_weights, dtype=np.float32),
    )


def split_clauses(text):
    clauses = [
        part.strip()
        for part in re.split(
            r"[.!?;,]+|\bwhile\b|\bthen\b|\bbut\b",
            text,
            flags=re.IGNORECASE,
        )
        if part.strip()
    ]
    if text not in clauses:
        clauses.insert(0, text)
    return clauses


def flatten_runtime_segments(texts):
    segments = []
    ranges = []
    for text in texts:
        start = len(segments)
        segments.extend(split_clauses(text))
        ranges.append((start, len(segments)))
    return segments, ranges


def tune_threshold(targets, probabilities):
    positives = int(targets.sum())
    negatives = int(len(targets) - positives)
    if positives < 10 or negatives < 10:
        return 0.5

    from sklearn.metrics import f1_score

    best_threshold = 0.5
    best_f1 = -1.0
    for threshold in np.linspace(0.05, 0.95, 91):
        score = f1_score(
            targets,
            probabilities >= threshold,
            zero_division=0,
        )
        if score > best_f1 or (
            score == best_f1
            and abs(threshold - 0.5) < abs(best_threshold - 0.5)
        ):
            best_f1 = score
            best_threshold = float(threshold)
    return round(best_threshold, 2)


def fit_binary_model(features, targets, sample_weights, max_iter, seed):
    unique = np.unique(targets)
    if len(unique) == 1:
        return {
            "type": "constant",
            "probability": float(unique[0]),
        }

    from sklearn.linear_model import LogisticRegression

    estimator = LogisticRegression(
        class_weight="balanced",
        max_iter=max_iter,
        random_state=seed,
        solver="lbfgs",
    )
    estimator.fit(features, targets, sample_weight=sample_weights)
    return {"type": "logistic_regression", "estimator": estimator}


def predict_label(model, features):
    if model["type"] == "constant":
        return np.full(
            features.shape[0],
            model["probability"],
            dtype=np.float32,
        )
    return model["estimator"].predict_proba(features)[:, 1]


def aggregate_runtime_probabilities(
    model,
    segment_features,
    segment_ranges,
    use_full_text_only=False,
):
    segment_probabilities = predict_label(model, segment_features)
    if use_full_text_only:
        return np.asarray(
            [segment_probabilities[start] for start, _ in segment_ranges],
            dtype=np.float32,
        )
    return np.asarray(
        [
            np.max(segment_probabilities[start:end])
            for start, end in segment_ranges
        ],
        dtype=np.float32,
    )


def runtime_predictions(probabilities, thresholds, labels):
    predictions = np.zeros(probabilities.shape, dtype=bool)
    neutral_index = labels.index("neutral")
    for row_index, row in enumerate(probabilities):
        detected = [
            index
            for index, probability in enumerate(row)
            if index != neutral_index and probability >= thresholds[index]
        ]
        neutral_active = row[neutral_index] >= thresholds[neutral_index]
        if neutral_active:
            detected = [
                index
                for index in detected
                if (
                    row[index] >= NEUTRAL_OVERRIDE_CONFIDENCE
                    or row[index]
                    >= row[neutral_index] + NEUTRAL_COMPETITION_MARGIN
                )
            ]
        detected.sort(key=lambda index: row[index], reverse=True)
        detected = detected[:MAX_RUNTIME_EMOTIONS]
        if detected:
            predictions[row_index, detected] = True
        elif neutral_active:
            predictions[row_index, neutral_index] = True
    return predictions


def evaluate(targets, probabilities, thresholds, labels):
    from sklearn.metrics import precision_recall_fscore_support

    predictions = runtime_predictions(
        probabilities,
        thresholds,
        labels,
    )
    metrics = {}
    for average in ("micro", "macro"):
        precision, recall, f1, _ = precision_recall_fscore_support(
            targets,
            predictions,
            average=average,
            zero_division=0,
        )
        metrics[average] = {
            "precision": round(float(precision), 4),
            "recall": round(float(recall), 4),
            "f1": round(float(f1), 4),
        }

    per_label = {}
    for index, label in enumerate(labels):
        precision, recall, f1, _ = precision_recall_fscore_support(
            targets[:, index],
            predictions[:, index],
            average="binary",
            zero_division=0,
        )
        per_label[label] = {
            "positives": int(targets[:, index].sum()),
            "predicted_positives": int(predictions[:, index].sum()),
            "threshold": float(thresholds[index]),
            "precision": round(float(precision), 4),
            "recall": round(float(recall), 4),
            "f1": round(float(f1), 4),
        }
    metrics["per_emotion"] = per_label
    return metrics


def main():
    args = parse_args()
    data_dir = args.data_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    schema_path = data_dir / "label_schema.json"
    required_paths = [
        schema_path,
        data_dir / "train.jsonl",
        data_dir / "validation.jsonl",
        data_dir / "test.jsonl",
    ]
    missing = [str(path) for path in required_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Build the dataset before training. Missing: "
            + ", ".join(missing)
        )

    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    labels = schema["emotion_labels"]
    split_limits = {
        "train": args.train_limit,
        "validation": args.validation_limit,
        "test": args.test_limit,
    }
    splits = {}
    for split, limit in split_limits.items():
        splits[split] = load_split(
            data_dir / f"{split}.jsonl",
            labels,
            limit,
        )
        print(f"Loaded {len(splits[split][0]):,} {split} examples")

    print(f"Loading sentence encoder: {args.encoder}", flush=True)
    from sentence_transformers import SentenceTransformer

    encoder = SentenceTransformer(args.encoder)
    print("Encoding training text...", flush=True)
    train_features = encoder.encode(
        splits["train"][0],
        batch_size=args.batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    ).astype(np.float32, copy=False)

    runtime_features = {}
    runtime_ranges = {}
    for split in ("validation", "test"):
        segments, ranges = flatten_runtime_segments(splits[split][0])
        print(
            f"Encoding {split} clauses "
            f"({len(segments):,} segments)...",
            flush=True,
        )
        runtime_features[split] = encoder.encode(
            segments,
            batch_size=args.batch_size,
            normalize_embeddings=True,
            show_progress_bar=True,
            convert_to_numpy=True,
        ).astype(np.float32, copy=False)
        runtime_ranges[split] = ranges

    train_targets = splits["train"][1]
    train_weights = splits["train"][2]
    validation_targets = splits["validation"][1]
    test_targets = splits["test"][1]

    emotion_classifiers = {}
    validation_probabilities = np.zeros(
        validation_targets.shape,
        dtype=np.float32,
    )
    test_probabilities = np.zeros(test_targets.shape, dtype=np.float32)
    thresholds = np.zeros(len(labels), dtype=np.float64)

    for index, label in enumerate(labels):
        positives = int(train_targets[:, index].sum())
        print(f"Training emotion {label}: {positives:,} positives", flush=True)
        model = fit_binary_model(
            train_features,
            train_targets[:, index],
            train_weights,
            args.max_iter,
            args.seed,
        )
        emotion_classifiers[label] = model
        validation_probabilities[:, index] = aggregate_runtime_probabilities(
            model,
            runtime_features["validation"],
            runtime_ranges["validation"],
            use_full_text_only=label == "neutral",
        )
        test_probabilities[:, index] = aggregate_runtime_probabilities(
            model,
            runtime_features["test"],
            runtime_ranges["test"],
            use_full_text_only=label == "neutral",
        )
        tuned_threshold = tune_threshold(
            validation_targets[:, index],
            validation_probabilities[:, index],
        )
        thresholds[index] = (
            tuned_threshold
            if label == "neutral"
            else max(tuned_threshold, RUNTIME_MIN_CONFIDENCE)
        )

    metrics = evaluate(
        test_targets,
        test_probabilities,
        thresholds,
        labels,
    )
    metadata = {
        "artifact_version": 2,
        "task": "multi_label_emotion_classification",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "encoder": args.encoder,
        "inference": {
            "clause_max": True,
            "runtime_min_confidence": RUNTIME_MIN_CONFIDENCE,
            "neutral_uses_full_text": True,
            "neutral_competition_margin": NEUTRAL_COMPETITION_MARGIN,
            "neutral_override_confidence": NEUTRAL_OVERRIDE_CONFIDENCE,
            "max_emotions": MAX_RUNTIME_EMOTIONS,
        },
        "labels": labels,
        "thresholds": {
            label: float(thresholds[index])
            for index, label in enumerate(labels)
        },
        "split_sizes": {
            split: len(values[0]) for split, values in splits.items()
        },
        "training_emotion_counts": dict(
            zip(labels, map(int, train_targets.sum(axis=0)))
        ),
        "training_emotion_combinations": dict(
            Counter(
                ",".join(
                    label
                    for label, active in zip(labels, row)
                    if active
                )
                or "neutral"
                for row in train_targets
            )
        ),
    }
    artifact = {
        "metadata": metadata,
        "emotion_classifiers": emotion_classifiers,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    from joblib import dump

    artifact_path = output_dir / "empwave_classifier.joblib"
    metrics_path = output_dir / "training_metrics.json"
    dump(artifact, artifact_path)
    metrics_path.write_text(
        json.dumps(
            {"metadata": metadata, "test_metrics": metrics},
            indent=2,
        ),
        encoding="utf-8",
    )

    print(json.dumps(metrics, indent=2))
    print(f"Saved model: {artifact_path}")
    print(f"Saved metrics: {metrics_path}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nTraining interrupted; no complete model was saved.")
        raise SystemExit(130)
