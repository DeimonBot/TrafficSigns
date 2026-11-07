"""Descarga previa de los modelos de Hugging Face usados por RoadVision."""
from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from huggingface_hub import list_repo_files, snapshot_download  # noqa: E402

from roadvision.config import Config  # noqa: E402

# Pesos que PyTorch no usa (TensorFlow, Flax, ONNX, TFLite...).
_IGNORAR = ["*.h5", "*.msgpack", "*.onnx", "*.tflite", "*.ot"]


def descargar(ids: Sequence[str]) -> None:
    for modelo_id in ids:
        ignorar = list(_IGNORAR)
        if "model.safetensors" in list_repo_files(modelo_id):
            ignorar.append("pytorch_model.bin")
        snapshot_download(modelo_id, ignore_patterns=ignorar)
        print(f"OK {modelo_id}")


def main() -> None:
    c = Config()
    descargar([c.modelo_detector, c.modelo_clasificador, c.modelo_respaldo])


if __name__ == "__main__":
    main()
