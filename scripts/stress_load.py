"""Generador de carga sintética para RoadVision.

Ejecuta multiplicaciones de matrices en ciclos de carga y reposo para que el
dashboard de métricas muestre el uso de CPU/GPU moviéndose, sin necesidad de
cargar los modelos reales. Usa GPU (torch + CUDA) si está disponible y, si no,
CPU con numpy o, como último recurso, un bucle Python puro.

Uso:
    python scripts/stress_load.py [--segundos 30] [--dispositivo auto|cpu|cuda] [--tamano 2048]

Autor: jhoan
Fecha: 2026-11-06
"""
from __future__ import annotations

import argparse
import random
import sys
import time
from collections.abc import Callable, Sequence
from typing import Any

CARGA_S = 2.0            # duración de cada fase de carga
REPOSO_S = 1.0           # duración de cada fase de reposo
TAMANO_PURO_PYTHON = 96  # tope de dimensión para el fallback sin numpy


def _importar_torch() -> Any | None:
    """Importa torch de forma perezosa; devuelve None si no está instalado."""
    try:
        import torch
    except ImportError:
        return None
    return torch


def _importar_numpy() -> Any | None:
    """Importa numpy de forma perezosa; devuelve None si no está instalado."""
    try:
        import numpy
    except ImportError:
        return None
    return numpy


def _preparar_gpu(torch: Any, tamano: int) -> tuple[str, Callable[[], None]]:
    a = torch.randn(tamano, tamano, device="cuda")
    b = torch.randn(tamano, tamano, device="cuda")

    def paso() -> None:
        torch.matmul(a, b)
        torch.cuda.synchronize()

    return f"GPU ({torch.cuda.get_device_name(0)})", paso


def _preparar_numpy(np: Any, tamano: int) -> tuple[str, Callable[[], None]]:
    a = np.random.rand(tamano, tamano).astype(np.float32)
    b = np.random.rand(tamano, tamano).astype(np.float32)

    def paso() -> None:
        a @ b

    return f"CPU (numpy {np.__version__}, {tamano}x{tamano})", paso


def _preparar_python(tamano: int) -> tuple[str, Callable[[], None]]:
    n = min(tamano, TAMANO_PURO_PYTHON)
    a = [[random.random() for _ in range(n)] for _ in range(n)]
    columnas = list(zip(*[[random.random() for _ in range(n)] for _ in range(n)]))

    def paso() -> None:
        for fila in a:
            [sum(x * y for x, y in zip(fila, col)) for col in columnas]

    return f"CPU (Python puro, {n}x{n})", paso


def _elegir_backend(dispositivo: str, tamano: int) -> tuple[str, Callable[[], None]]:
    """Elige el backend de carga según el pedido y lo disponible en el equipo."""
    if dispositivo in ("auto", "cuda"):
        torch = _importar_torch()
        if torch is not None and torch.cuda.is_available():
            try:
                return _preparar_gpu(torch, tamano)
            except RuntimeError as error:
                print(f"AVISO: no se pudo preparar la GPU ({error}); se usa CPU.", flush=True)
        elif dispositivo == "cuda":
            print("AVISO: CUDA no disponible (torch ausente o sin GPU); se usa CPU.", flush=True)

    np = _importar_numpy()
    if np is not None:
        return _preparar_numpy(np, tamano)

    print("AVISO: numpy no instalado; se usa bucle Python puro (carga más ligera).", flush=True)
    return _preparar_python(tamano)


def _bucle_carga(etiqueta: str, paso: Callable[[], None], segundos: float) -> None:
    """Alterna fases de carga y reposo hasta cumplir la duración pedida."""
    inicio = time.monotonic()
    ciclo = 0

    def transcurrido() -> float:
        return time.monotonic() - inicio

    while transcurrido() < segundos:
        ciclo += 1
        fin_carga = min(time.monotonic() + CARGA_S, inicio + segundos)
        print(f"[ciclo {ciclo:03d}] CARGA  | {etiqueta} | t={transcurrido():6.1f}s / {segundos:g}s",
              flush=True)
        while time.monotonic() < fin_carga:
            paso()

        if transcurrido() >= segundos:
            break

        fin_reposo = min(time.monotonic() + REPOSO_S, inicio + segundos)
        print(f"[ciclo {ciclo:03d}] REPOSO | {etiqueta} | t={transcurrido():6.1f}s / {segundos:g}s",
              flush=True)
        time.sleep(max(0.0, fin_reposo - time.monotonic()))

    print(f"Terminado: {ciclo} ciclos en {transcurrido():.1f}s.", flush=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generador de carga sintética para RoadVision.")
    parser.add_argument("--segundos", type=float, default=30.0, help="duración total en segundos")
    parser.add_argument("--dispositivo", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--tamano", type=int, default=2048, help="dimensión de las matrices")
    args = parser.parse_args(argv)

    if args.segundos <= 0:
        parser.error("--segundos debe ser mayor que 0")
    if args.tamano < 8:
        parser.error("--tamano debe ser al menos 8")

    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except (OSError, ValueError):
            pass

    etiqueta, paso = _elegir_backend(args.dispositivo, args.tamano)
    print(f"stress_load: backend={etiqueta}, duración={args.segundos:g}s. Ctrl+C para terminar.",
          flush=True)

    try:
        _bucle_carga(etiqueta, paso, args.segundos)
    except KeyboardInterrupt:
        print("\nDetenido por el usuario (Ctrl+C).", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
