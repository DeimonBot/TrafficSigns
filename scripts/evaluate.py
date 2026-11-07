"""Evalúa los modelos de RoadVision. Nivel A: exactitud del clasificador en GTSRB."""
from __future__ import annotations

import argparse
import sys
from collections.abc import Iterator, Sequence
from itertools import islice
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image  # noqa: E402

from roadvision.classifier import ClasificadorGtsrb  # noqa: E402
from roadvision.config import Config, ruta_datasets  # noqa: E402
from roadvision.device import elegir_dispositivo  # noqa: E402
from roadvision.evaluacion import evaluar_clasificador  # noqa: E402


def _muestras(limite: int | None) -> Iterator[tuple[Image.Image, int]]:
    from torchvision.datasets import GTSRB

    datos = GTSRB(root=ruta_datasets(), split="test", download=True)
    for imagen, clase in islice(datos, limite):
        yield imagen.convert("RGB"), clase


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluación de RoadVision.")
    parser.add_argument("--nivel", choices=["a"], required=True)
    parser.add_argument("--limite", type=int, default=None, help="máximo de imágenes")
    parser.add_argument("--dispositivo", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--sin-fp16", action="store_true", help="desactiva media precisión")
    args = parser.parse_args(argv)

    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except (OSError, ValueError):
            pass

    dispositivo = elegir_dispositivo(args.dispositivo, fp16=not args.sin_fp16)
    clasificador = ClasificadorGtsrb(Config().modelo_clasificador, dispositivo)
    reporte = evaluar_clasificador(clasificador, _muestras(args.limite))

    print(f"n: {reporte.n}")
    print(f"exactitud: {reporte.exactitud:.4f}")
    print("5 etiquetas con peor exactitud:")
    peores = sorted(reporte.exactitud_por_etiqueta.items(), key=lambda kv: kv[1])[:5]
    for etiqueta, valor in peores:
        print(f"  {valor:.4f}  {etiqueta}")
    print(f"imágenes por segundo: {reporte.imagenes_por_segundo:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
