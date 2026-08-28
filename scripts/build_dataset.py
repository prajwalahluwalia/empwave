#!/usr/bin/env python3
import argparse
import csv
import hashlib
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from empwave.services.intent_classifier import (
    ACTIVATION_STAGE,
    INTENT_LABELS,
    SemanticIntentClassifier,
    REGION_CONFIG,
    SPECIAL_INTENTS,
)


GOEMOTIONS_SPLITS = {
    "train": "train.tsv",
    "validation": "dev.tsv",
    "test": "test.tsv",
}
MULTINLI_SPLITS = {
    "validation": "validation_matched.csv",
    "test": "validation_mismatched.csv",
    "train": "train.csv",
}
NLI_LABELS = {
    "0": "entailment",
    "1": "neutral",
    "2": "contradiction",
}
COGNITIVE_EMOTIONS = {"confusion", "curiosity", "realization"}


def normalize_text(text):
    return re.sub(r"\s+", " ", text).strip()


def text_fingerprint(text):
    return hashlib.sha256(normalize_text(text).lower().encode("utf-8")).hexdigest()


class SplitDeduplicator:
    def __init__(self):
        self.seen = set()

    def accept(self, text):
        fingerprint = text_fingerprint(text)
        if fingerprint in self.seen:
            return False
        self.seen.add(fingerprint)
        return True


class ReservoirBuckets:
    def __init__(self, quota, seed):
        self.quota = quota
        self.random = random.Random(seed)
        self.seen = Counter()
        self.buckets = defaultdict(list)

    def add(self, bucket, record):
        self.seen[bucket] += 1
        values = self.buckets[bucket]
        if len(values) < self.quota:
            values.append(record)
            return

        replacement = self.random.randrange(self.seen[bucket])
        if replacement < self.quota:
            values[replacement] = record

    def records(self):
        return [
            record
            for bucket in sorted(self.buckets)
            for record in self.buckets[bucket]
        ]


def load_goemotions_labels(source_root):
    path = source_root / "goemotions" / "data" / "emotions.txt"
    return path.read_text(encoding="utf-8").splitlines()


def build_goemotions_records(source_root, deduplicator):
    labels = load_goemotions_labels(source_root)
    records = defaultdict(list)

    for split, filename in GOEMOTIONS_SPLITS.items():
        path = source_root / "goemotions" / "data" / filename
        with path.open(encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                text, raw_label_ids, source_id = line.rstrip("\n").split("\t")
                text = normalize_text(text)
                if not text or not deduplicator.accept(text):
                    continue

                emotion_labels = [
                    labels[int(label_id)]
                    for label_id in raw_label_ids.split(",")
                ]
                non_neutral = [
                    label for label in emotion_labels if label != "neutral"
                ]
                regions = []
                if non_neutral:
                    regions.append("amygdala")
                if COGNITIVE_EMOTIONS.intersection(non_neutral):
                    regions.append("prefrontal")

                primary_intent = (
                    "Reasoning and evaluation"
                    if "prefrontal" in regions
                    else "Emotional significance"
                    if "amygdala" in regions
                    else "Neutral statement"
                )
                records[split].append(
                    {
                        "id": (
                            f"goemotions:{split}:{source_id}:{line_number}"
                        ),
                        "text": text,
                        "emotions": emotion_labels,
                        "intents": [],
                        "regions": regions,
                        "primary_intent": primary_intent,
                        "label_source": "human_annotation_mapped",
                        "sample_weight": 1.0,
                        "source": {
                            "dataset": "GoEmotions",
                            "split": split,
                            "source_id": source_id,
                            "emotion_labels": emotion_labels,
                            "line": line_number,
                        },
                    }
                )

    return records


def semantic_batch_labels(texts, classifier, threshold_margin):
    embeddings = classifier.model.encode(
        texts,
        batch_size=256,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    scores_by_region = {}
    concepts_by_region = {}

    for region_id, config in REGION_CONFIG.items():
        similarities = embeddings @ classifier.prototype_embeddings[region_id].T
        best_indices = np.argmax(similarities, axis=1)
        scores_by_region[region_id] = similarities[
            np.arange(len(texts)), best_indices
        ]
        concepts_by_region[region_id] = [
            config["prototypes"][int(index)] for index in best_indices
        ]

    special_matches = []
    for intent_id, config in SPECIAL_INTENTS.items():
        similarities = (
            embeddings @ classifier.special_intent_embeddings[intent_id].T
        )
        best_indices = np.argmax(similarities, axis=1)
        special_matches.append(
            (
                config,
                similarities[np.arange(len(texts)), best_indices],
                [
                    config["prototypes"][int(index)]
                    for index in best_indices
                ],
            )
        )

    results = []
    for row_index, text in enumerate(texts):
        selected = {}
        for region_id, config in REGION_CONFIG.items():
            score = float(scores_by_region[region_id][row_index])
            if score < config["threshold"] + threshold_margin:
                continue
            selected[region_id] = {
                "id": region_id,
                "score": score,
                "intent": INTENT_LABELS[region_id],
                "semantic_concept": concepts_by_region[region_id][row_index],
            }

        for config, scores, concepts in special_matches:
            score = float(scores[row_index])
            if score < config["threshold"] + threshold_margin:
                continue
            region_id = config["region_id"]
            if region_id in selected and selected[region_id]["score"] >= score:
                continue
            selected[region_id] = {
                "id": region_id,
                "score": score,
                "intent": config["label"],
                "semantic_concept": concepts[row_index],
            }

        prefrontal = selected.get("prefrontal")
        cerebellum = selected.get("cerebellum")
        if prefrontal and cerebellum:
            loser = (
                "cerebellum"
                if prefrontal["score"] >= cerebellum["score"]
                else "prefrontal"
            )
            selected.pop(loser)

        ordered = sorted(
            selected.values(),
            key=lambda item: (
                ACTIVATION_STAGE[item["id"]],
                -item["score"],
            ),
        )
        primary = max(ordered, key=lambda item: item["score"]) if ordered else None
        results.append(
            {
                "text": text,
                "regions": [item["id"] for item in ordered],
                "primary_intent": (
                    primary["intent"] if primary else "Neutral statement"
                ),
                "semantic_scores": {
                    item["id"]: round(item["score"], 4) for item in ordered
                },
                "semantic_concepts": {
                    item["id"]: item["semantic_concept"] for item in ordered
                },
            }
        )

    return results


def build_multinli_records(
    source_root,
    deduplicator,
    classifier,
    quota,
    threshold_margin,
    batch_size,
    max_train_rows,
    seed,
):
    records = defaultdict(list)

    for split, filename in MULTINLI_SPLITS.items():
        path = source_root / "multinli" / filename
        reservoir = ReservoirBuckets(quota=quota, seed=seed)
        pending_texts = []
        pending_metadata = []
        processed = 0

        def flush():
            if not pending_texts:
                return
            labels = semantic_batch_labels(
                pending_texts,
                classifier,
                threshold_margin,
            )
            for label, metadata in zip(labels, pending_metadata):
                regions = label["regions"]
                bucket = (
                    max(
                        label["semantic_scores"],
                        key=label["semantic_scores"].get,
                    )
                    if regions
                    else "neutral"
                )
                reservoir.add(
                    bucket,
                    {
                        "id": (
                            f"multinli:{split}:{metadata['pair_id']}:"
                            f"{metadata['line']}:hypothesis"
                        ),
                        "text": label["text"],
                        "emotions": [],
                        "intents": regions,
                        "regions": regions,
                        "primary_intent": label["primary_intent"],
                        "label_source": "semantic_weak_label",
                        "sample_weight": 0.45,
                        "semantic_scores": label["semantic_scores"],
                        "semantic_concepts": label["semantic_concepts"],
                        "source": {
                            "dataset": "MultiNLI",
                            "split": split,
                            "pair_id": metadata["pair_id"],
                            "line": metadata["line"],
                            "genre": metadata["genre"],
                            "nli_relation": metadata["nli_relation"],
                        },
                    },
                )
            pending_texts.clear()
            pending_metadata.clear()

        with path.open(newline="", encoding="utf-8") as source:
            reader = csv.DictReader(source)
            for row in reader:
                if split == "train" and processed >= max_train_rows:
                    break
                processed += 1

                text = normalize_text(row["hypothesis"])
                if not text or len(text) > 1000 or not deduplicator.accept(text):
                    continue
                pending_texts.append(text)
                pending_metadata.append(
                    {
                        "pair_id": row["pairID"],
                        "line": processed + 1,
                        "genre": row["genre"],
                        "nli_relation": NLI_LABELS.get(
                            row["label"], row["label"]
                        ),
                    }
                )
                if len(pending_texts) >= batch_size:
                    flush()
                    if processed % 20_000 == 0:
                        print(f"Processed {processed:,} rows from {filename}")
            flush()

        records[split].extend(reservoir.records())
        print(
            f"Selected {len(records[split]):,} balanced weak labels "
            f"from {filename}"
        )

    return records


def write_jsonl(path, records):
    with path.open("w", encoding="utf-8") as destination:
        for record in records:
            destination.write(json.dumps(record, ensure_ascii=False) + "\n")


def summarize(records):
    return {
        split: {
            "examples": len(values),
            "sources": dict(
                Counter(record["source"]["dataset"] for record in values)
            ),
            "label_sources": dict(
                Counter(record["label_source"] for record in values)
            ),
            "region_counts": dict(
                Counter(
                    region
                    for record in values
                    for region in record["regions"]
                )
            ),
            "emotion_counts": dict(
                Counter(
                    emotion
                    for record in values
                    for emotion in record["emotions"]
                )
            ),
            "intent_counts": dict(
                Counter(
                    intent
                    for record in values
                    for intent in record["intents"]
                )
            ),
            "neutral_examples": sum(
                not record["regions"] for record in values
            ),
        }
        for split, values in records.items()
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build the Empwave weakly supervised brain-intent dataset."
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path.home() / "Desktop" / "data",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data") / "processed",
    )
    parser.add_argument("--multinli-quota", type=int, default=2000)
    parser.add_argument("--threshold-margin", type=float, default=0.08)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-train-rows", type=int, default=150_000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    source_root = args.source_root.expanduser().resolve()
    goemotions_path = source_root / "goemotions" / "data" / "train.tsv"
    multinli_path = source_root / "multinli" / "train.csv"
    if not goemotions_path.is_file() or not multinli_path.is_file():
        raise FileNotFoundError(
            "Expected GoEmotions and MultiNLI under "
            f"{source_root}. Use --source-root to specify another location."
        )

    csv.field_size_limit(sys.maxsize)
    deduplicator = SplitDeduplicator()

    goemotions = build_goemotions_records(source_root, deduplicator)
    classifier = SemanticIntentClassifier()
    multinli = build_multinli_records(
        source_root=source_root,
        deduplicator=deduplicator,
        classifier=classifier,
        quota=args.multinli_quota,
        threshold_margin=args.threshold_margin,
        batch_size=args.batch_size,
        max_train_rows=args.max_train_rows,
        seed=args.seed,
    )

    combined = {}
    for split in ("train", "validation", "test"):
        combined[split] = goemotions[split] + multinli[split]
        random.Random(args.seed).shuffle(combined[split])

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    for split, values in combined.items():
        write_jsonl(output_dir / f"{split}.jsonl", values)

    schema = {
        "format": "JSON Lines",
        "task": (
            "multi-label emotion and semantic-intent classification "
            "for illustrative brain-region mapping"
        ),
        "emotion_labels": load_goemotions_labels(source_root),
        "intent_labels": list(REGION_CONFIG),
        "contextual_regions": list(REGION_CONFIG),
        "automatic_runtime_baseline": ["temporal_l"],
        "fields": {
            "id": "Stable source-derived identifier",
            "text": "Input utterance",
            "emotions": "Original human-annotated emotion labels",
            "intents": "Semantic weak intent labels",
            "regions": "Legacy derived labels retained for compatibility",
            "primary_intent": "Human-readable primary intent",
            "label_source": "Human mapping or semantic weak label",
            "sample_weight": "Recommended training loss weight",
            "source": "Dataset provenance",
        },
    }
    (output_dir / "label_schema.json").write_text(
        json.dumps(schema, indent=2),
        encoding="utf-8",
    )
    stats = summarize(combined)
    (output_dir / "dataset_stats.json").write_text(
        json.dumps(stats, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
