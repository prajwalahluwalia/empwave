"""Memory-efficient MiniLM sentence encoding for Empwave inference."""

import os
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNTIME_ENCODER_PATH = (
    PROJECT_ROOT / "models" / "runtime" / "all-MiniLM-L6-v2-onnx"
)


class TextEncoder:
    """Expose the SentenceTransformer encode interface used by Empwave."""

    def __init__(self, model_name):
        configured_path = os.getenv("EMPWAVE_ENCODER_PATH")
        runtime_path = (
            Path(configured_path)
            if configured_path
            else RUNTIME_ENCODER_PATH
        )
        self.model_name = model_name
        self.runtime_model_path = runtime_path / "model.int8.onnx"
        if self.runtime_model_path.is_file():
            self._initialize_onnx(runtime_path)
        else:
            self._initialize_pytorch(model_name)

    def _initialize_onnx(self, runtime_path):
        import onnxruntime as ort
        from tokenizers import Tokenizer

        self.backend = "onnx"
        self.tokenizer = Tokenizer.from_file(
            str(runtime_path / "tokenizer.json")
        )
        self.tokenizer.enable_truncation(max_length=256)
        self.tokenizer.enable_padding(
            pad_id=self.tokenizer.token_to_id("[PAD]") or 0,
            pad_token="[PAD]",
        )
        options = ort.SessionOptions()
        options.intra_op_num_threads = int(os.getenv("OMP_NUM_THREADS", "1"))
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )
        self.session = ort.InferenceSession(
            str(self.runtime_model_path),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        self.input_names = {
            model_input.name for model_input in self.session.get_inputs()
        }

    def _initialize_pytorch(self, model_name):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.backend = "pytorch"
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()

    def _encode_onnx(self, texts, normalize_embeddings):
        encodings = self.tokenizer.encode_batch(texts)
        inputs = {
            "input_ids": np.asarray(
                [encoding.ids for encoding in encodings],
                dtype=np.int64,
            ),
            "attention_mask": np.asarray(
                [encoding.attention_mask for encoding in encodings],
                dtype=np.int64,
            ),
            "token_type_ids": np.asarray(
                [encoding.type_ids for encoding in encodings],
                dtype=np.int64,
            ),
        }
        embeddings = self.session.run(
            None,
            {
                name: value
                for name, value in inputs.items()
                if name in self.input_names
            },
        )[0].astype(np.float32, copy=False)
        if normalize_embeddings:
            norms = np.linalg.norm(
                embeddings,
                axis=1,
                keepdims=True,
            )
            embeddings = embeddings / np.maximum(norms, 1e-12)
        return embeddings

    def _encode_pytorch(self, texts, normalize_embeddings):
        batches = []
        with self.torch.inference_mode():
            for start in range(0, len(texts), 64):
                encoded = self.tokenizer(
                    texts[start:start + 64],
                    padding=True,
                    truncation=True,
                    max_length=256,
                    return_tensors="pt",
                )
                hidden = self.model(**encoded).last_hidden_state
                attention_mask = encoded["attention_mask"].unsqueeze(-1)
                embeddings = (
                    (hidden * attention_mask).sum(dim=1)
                    / attention_mask.sum(dim=1).clamp(min=1)
                )
                if normalize_embeddings:
                    embeddings = self.torch.nn.functional.normalize(
                        embeddings,
                        p=2,
                        dim=1,
                    )
                batches.append(embeddings.float().cpu().numpy())
        return np.concatenate(batches, axis=0).astype(
            np.float32,
            copy=False,
        )

    def encode(
        self,
        sentences,
        normalize_embeddings=False,
        show_progress_bar=False,
        convert_to_numpy=True,
    ):
        del show_progress_bar
        single_sentence = isinstance(sentences, str)
        texts = [sentences] if single_sentence else list(sentences)
        if not texts:
            result = np.empty((0, 384), dtype=np.float32)
        elif self.backend == "onnx":
            result = self._encode_onnx(texts, normalize_embeddings)
        else:
            result = self._encode_pytorch(texts, normalize_embeddings)
        if not convert_to_numpy:
            if self.backend == "pytorch":
                result = self.torch.from_numpy(result)
            else:
                raise ValueError(
                    "ONNX inference only supports NumPy encoder output."
                )
        return result[0] if single_sentence else result
