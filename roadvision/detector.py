"""Detector genérico de señales de tránsito con OWLv2 (vocabulario abierto)."""
from __future__ import annotations

from collections.abc import Sequence

import torch
from PIL import Image
from transformers import Owlv2ForObjectDetection, Owlv2Processor

from roadvision.contracts import Caja, Deteccion
from roadvision.device import Dispositivo


class DetectorOwlv2:
    def __init__(self, modelo_id: str, consultas: Sequence[str], dispositivo: Dispositivo) -> None:
        self._dispositivo = dispositivo
        self._consultas = list(consultas)
        self._procesador = Owlv2Processor.from_pretrained(modelo_id)
        self._modelo = (
            Owlv2ForObjectDetection.from_pretrained(modelo_id)
            .to(device=dispositivo.nombre, dtype=dispositivo.dtype)
            .eval()
        )

    def detectar(self, imagen: Image.Image, umbral: float) -> list[Deteccion]:
        imagen = imagen.convert("RGB")
        ancho, alto = imagen.size
        entradas = self._procesador(text=[self._consultas], images=imagen, return_tensors="pt")
        entradas = entradas.to(self._dispositivo.nombre)
        entradas["pixel_values"] = entradas["pixel_values"].to(dtype=self._dispositivo.dtype)
        with torch.inference_mode():
            salidas = self._modelo(**entradas)
        # El post-proceso ya compensa el relleno a cuadrado: se pasa el tamaño original.
        resultado = self._procesador.post_process_grounded_object_detection(
            salidas, threshold=umbral, target_sizes=[(alto, ancho)]
        )[0]
        detecciones: list[Deteccion] = []
        for caja, puntaje in zip(resultado["boxes"].float().tolist(), resultado["scores"].float().tolist()):
            x1 = min(max(round(caja[0]), 0), ancho)
            y1 = min(max(round(caja[1]), 0), alto)
            x2 = min(max(round(caja[2]), 0), ancho)
            y2 = min(max(round(caja[3]), 0), alto)
            if x2 <= x1 or y2 <= y1:
                continue
            detecciones.append(Deteccion(caja=(x1, y1, x2, y2), puntaje=float(puntaje)))
        return detecciones
