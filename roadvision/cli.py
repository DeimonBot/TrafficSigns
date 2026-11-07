"""Comando de línea: `python -m roadvision.cli foto.jpg`."""
from __future__ import annotations

import argparse
import dataclasses
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from roadvision.config import Config
from roadvision.contracts import Resultado
from roadvision.pipeline import ETAPAS, Pipeline, construir_pipeline

CALENTAMIENTO = ("deteccion", "clasificacion", "respaldo")


def _formatear(resultado: Resultado) -> str:
    lineas = []
    for s in resultado.senales:
        categoria = s.categoria.value if s.categoria is not None else "sin categoría"
        lineas.append(
            f"#{s.numero} {s.nombre} [{categoria}] {s.estado.value} conf {s.confianza:.2f}"
        )
        if s.significado:
            lineas.append(f"    {s.significado}")
    lineas.append(f"Resumen: {resultado.conteo_por_nombre}")
    tiempos = " ".join(
        f"{e}={resultado.tiempos_ms.get(e, 0.0):.0f}" for e in ETAPAS
    )
    lineas.append(f"Tiempos (ms): {tiempos}")
    return "\n".join(lineas)


def main(
    argv: Sequence[str] | None = None,
    construir: Callable[[Config], Pipeline] = construir_pipeline,
) -> int:
    parser = argparse.ArgumentParser(
        prog="roadvision", description="Analiza las señales de tránsito de una foto."
    )
    parser.add_argument("imagen", help="ruta de la foto")
    parser.add_argument("--dispositivo", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--sin-fp16", action="store_true", help="desactiva media precisión")
    args = parser.parse_args(argv)

    ruta = Path(args.imagen)
    if not ruta.is_file():
        print(f"No existe el archivo: {args.imagen}", file=sys.stderr)
        return 2
    try:
        imagen = Image.open(ruta)
        imagen.load()
    except (UnidentifiedImageError, OSError):
        print(f"No se pudo abrir la imagen: {args.imagen}", file=sys.stderr)
        return 2

    config = dataclasses.replace(
        Config(), dispositivo=args.dispositivo, fp16=not args.sin_fp16
    )
    pipeline = construir(config)
    calentamiento = pipeline.calentar()
    resultado = pipeline.analizar(imagen)

    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except (OSError, ValueError):
            pass
    print(
        "Calentamiento (ms, no incluido en los tiempos): "
        + " ".join(f"{e}={calentamiento.get(e, 0.0):.1f}" for e in CALENTAMIENTO)
    )
    print(_formatear(resultado))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
