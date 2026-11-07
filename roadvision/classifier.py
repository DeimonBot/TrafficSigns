"""Clasificador ViT de señales GTSRB (43 clases)."""
from __future__ import annotations

import torch
from PIL import Image
from transformers import AutoModelForImageClassification, ViTImageProcessor

from roadvision.contracts import Prediccion
from roadvision.device import Dispositivo


class ClasificadorGtsrb:
    def __init__(self, modelo_id: str, dispositivo: Dispositivo) -> None:
        self._dispositivo = dispositivo
        # El preprocessor_config.json declara ViTFeatureExtractor, que AutoImageProcessor
        # ya no reconoce; se usa ViTImageProcessor con los mismos valores (224x224, 0.5/0.5).
        self._procesador = ViTImageProcessor(
            size={"height": 224, "width": 224},
            image_mean=[0.5, 0.5, 0.5],
            image_std=[0.5, 0.5, 0.5],
        )
        self._modelo = (
            AutoModelForImageClassification.from_pretrained(modelo_id)
            .to(device=dispositivo.nombre, dtype=dispositivo.dtype)
            .eval()
        )
        id2label = self._modelo.config.id2label
        etiquetas = [id2label.get(i) for i in range(len(id2label))]
        if any(e is None for e in etiquetas):
            raise ValueError(f"El modelo {modelo_id} tiene etiquetas sin definir en id2label")
        self.etiquetas: list[str] = etiquetas

    def clasificar(self, recortes: list[Image.Image], tamano_lote: int = 32) -> list[Prediccion]:
        predicciones: list[Prediccion] = []
        for inicio in range(0, len(recortes), tamano_lote):
            lote = [r.convert("RGB") for r in recortes[inicio : inicio + tamano_lote]]
            entradas = self._procesador(images=lote, return_tensors="pt")
            pixeles = entradas["pixel_values"].to(
                device=self._dispositivo.nombre, dtype=self._dispositivo.dtype
            )
            with torch.inference_mode():
                logits = self._modelo(pixel_values=pixeles).logits
            probabilidades = torch.softmax(logits.float(), dim=-1)
            mejores, indices = probabilidades.max(dim=-1)
            for p, i in zip(mejores.tolist(), indices.tolist()):
                predicciones.append(Prediccion(etiqueta=self.etiquetas[i], probabilidad=p))
        return predicciones
