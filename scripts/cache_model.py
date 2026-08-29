#!/usr/bin/env python3
"""Create the low-memory ONNX sentence encoder used in deployment."""

import json
from pathlib import Path

import numpy as np
import torch
from joblib import load
from onnxruntime.quantization import QuantType, quantize_dynamic
from transformers import AutoModel, AutoTokenizer


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TARGET_PATH = (
    Path(__file__).resolve().parents[1]
    / "models"
    / "runtime"
    / "all-MiniLM-L6-v2-onnx"
)
TRAINED_CLASSIFIER_PATH = (
    Path(__file__).resolve().parents[1]
    / "models"
    / "trained"
    / "empwave_classifier.joblib"
)
RUNTIME_CLASSIFIER_PATH = (
    Path(__file__).resolve().parents[1]
    / "models"
    / "runtime"
    / "empwave_classifier.npz"
)


class MeanPoolingEncoder(torch.nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, input_ids, attention_mask, token_type_ids):
        hidden = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=False,
        )[0]
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)


TARGET_PATH.mkdir(parents=True, exist_ok=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
encoder = AutoModel.from_pretrained(MODEL_NAME)
model = MeanPoolingEncoder(encoder).eval()
tokenizer.save_pretrained(TARGET_PATH)
sample = tokenizer(
    ["Empwave runtime export"],
    padding=True,
    truncation=True,
    max_length=256,
    return_tensors="pt",
)
fp32_path = TARGET_PATH / "model.fp32.onnx"
int8_path = TARGET_PATH / "model.int8.onnx"
torch.onnx.export(
    model,
    (
        sample["input_ids"],
        sample["attention_mask"],
        sample["token_type_ids"],
    ),
    fp32_path,
    input_names=["input_ids", "attention_mask", "token_type_ids"],
    output_names=["sentence_embedding"],
    dynamic_axes={
        "input_ids": {0: "batch", 1: "sequence"},
        "attention_mask": {0: "batch", 1: "sequence"},
        "token_type_ids": {0: "batch", 1: "sequence"},
        "sentence_embedding": {0: "batch"},
    },
    opset_version=17,
    dynamo=False,
)
quantize_dynamic(
    fp32_path,
    int8_path,
    weight_type=QuantType.QInt8,
    per_channel=True,
)
fp32_path.unlink()
print(f"Cached INT8 ONNX {MODEL_NAME} at {int8_path}")

artifact = load(TRAINED_CLASSIFIER_PATH)
metadata = artifact["metadata"]
labels = metadata["labels"]
model_types = []
coefficients = np.zeros((len(labels), 384), dtype=np.float64)
intercepts = np.zeros(len(labels), dtype=np.float64)
constants = np.zeros(len(labels), dtype=np.float64)

for index, label in enumerate(labels):
    classifier = artifact["emotion_classifiers"][label]
    if classifier.get("type") == "constant":
        model_types.append("constant")
        constants[index] = float(classifier["probability"])
        continue

    estimator = classifier["estimator"]
    if estimator.classes_.tolist() != [0, 1]:
        raise ValueError(
            f"Unsupported classifier classes for {label}: "
            f"{estimator.classes_.tolist()}"
        )
    model_types.append("linear")
    coefficients[index] = estimator.coef_[0]
    intercepts[index] = estimator.intercept_[0]

RUNTIME_CLASSIFIER_PATH.parent.mkdir(parents=True, exist_ok=True)
np.savez_compressed(
    RUNTIME_CLASSIFIER_PATH,
    metadata_json=np.array(json.dumps(metadata)),
    labels=np.array(labels),
    model_types=np.array(model_types),
    coefficients=coefficients,
    intercepts=intercepts,
    constants=constants,
)
print(f"Cached NumPy emotion classifier at {RUNTIME_CLASSIFIER_PATH}")
