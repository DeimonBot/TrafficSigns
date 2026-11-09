"""Descarga SigLIP (capa segura) en la revisión fijada en data/capa_segura/meta.json.

Uso:  python scripts/descargar_siglip.py [--fijar]
  --fijar  consulta el commit actual del repositorio en Hugging Face y lo escribe en meta.json
           (solo para actualizar la versión; el flujo normal usa la revisión ya fijada).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from huggingface_hub import HfApi, list_repo_files, snapshot_download  # noqa: E402

from roadvision.capa_segura import RUTA_CAPA  # noqa: E402

# Pesos que PyTorch no usa (TensorFlow, Flax, ONNX, TFLite...), como en download_models.py.
_IGNORAR = ["*.h5", "*.msgpack", "*.onnx", "*.tflite", "*.ot"]


def main() -> None:
    ruta = RUTA_CAPA / "meta.json"
    meta = json.loads(ruta.read_text(encoding="utf-8"))
    modelo_id = meta["encoder"]
    if "--fijar" in sys.argv or not meta.get("revision_encoder"):
        meta["revision_encoder"] = HfApi().model_info(modelo_id).sha
        ruta.write_text(json.dumps(meta, indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"Revisión fijada en meta.json: {meta['revision_encoder']}")
    revision = meta["revision_encoder"]
    ignorar = list(_IGNORAR)
    if "model.safetensors" in list_repo_files(modelo_id, revision=revision):
        ignorar.append("pytorch_model.bin")
    snapshot_download(modelo_id, revision=revision, ignore_patterns=ignorar)
    print(f"OK {modelo_id}@{revision}")


if __name__ == "__main__":
    main()
