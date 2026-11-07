"""Selección de dispositivo (CUDA/CPU) y precisión."""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Dispositivo:
    nombre: str
    dtype: torch.dtype
    descripcion: str


def elegir_dispositivo(preferido: str = "auto", fp16: bool = True) -> Dispositivo:
    if preferido not in ("auto", "cuda", "cpu"):
        raise ValueError(
            f"Dispositivo no válido: {preferido!r} (use 'auto', 'cuda' o 'cpu')"
        )
    hay_cuda = torch.cuda.is_available()
    if preferido == "cuda" and not hay_cuda:
        raise RuntimeError(
            "Se pidió CUDA pero no hay una GPU NVIDIA con CUDA disponible"
        )
    if preferido == "cpu" or not hay_cuda:
        return Dispositivo("cpu", torch.float32, "CPU")
    dtype = torch.float16 if fp16 else torch.float32
    return Dispositivo("cuda", dtype, torch.cuda.get_device_name(0))


def sincronizar(dispositivo: Dispositivo) -> None:
    if dispositivo.nombre == "cuda":
        torch.cuda.synchronize()
