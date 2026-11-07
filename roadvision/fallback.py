"""Respaldo zero-shot con CLIP para verificar límites de velocidad y señales fuera de GTSRB."""
from __future__ import annotations

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from roadvision.device import Dispositivo


class RespaldoClip:
    def __init__(self, modelo_id: str, dispositivo: Dispositivo) -> None:
        self._dispositivo = dispositivo
        self._procesador = CLIPProcessor.from_pretrained(modelo_id)
        self._modelo = (
            CLIPModel.from_pretrained(modelo_id)
            .to(device=dispositivo.nombre, dtype=dispositivo.dtype)
            .eval()
        )

    def puntuar(self, recortes: list[Image.Image], textos: list[str]) -> list[list[float]]:
        if not textos:
            raise ValueError("Se requiere al menos un texto para puntuar")
        if not recortes:
            return []
        entradas = self._procesador(
            text=textos,
            images=[r.convert("RGB") for r in recortes],
            return_tensors="pt",
            padding=True,
        )
        entradas = {
            clave: (
                valor.to(device=self._dispositivo.nombre, dtype=self._dispositivo.dtype)
                if clave == "pixel_values"
                else valor.to(self._dispositivo.nombre)
            )
            for clave, valor in entradas.items()
        }
        with torch.inference_mode():
            logits = self._modelo(**entradas).logits_per_image
        return torch.softmax(logits.float(), dim=-1).tolist()
