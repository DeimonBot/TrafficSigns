"""Benchmark de RoadVision (spec §5.6): promedio y p95 por etapa, fuera del calentamiento.

Recorre una carpeta de fotos, repite el análisis N veces por imagen y reporta, por etapa
(total, preproceso, detección, clasificación, respaldo), n, media, p50, p95 y mín/máx, además
de picos de GPU/VRAM/temperatura, potencia promedio y CPU/RAM del proceso. El calentamiento
se mide UNA vez, se imprime aparte y nunca entra en las estadísticas. Cada fila se agrega al
CSV de métricas (el dashboard la muestra) salvo `--sin-csv`.

Uso:
    python scripts/benchmark.py [--carpeta samples] [--repeticiones 5]
        [--dispositivo auto|cpu|cuda] [--sin-fp16] [--csv RUTA | --sin-csv]
        [--comparar-precision] [--pausa-s SEGUNDOS]

Notas:
- Sin fotos en la carpeta se usan 3 imágenes SINTÉTICAS (ruido): no habrá señales detectadas;
  para resultados reales hay que poner fotos reales (jpg/jpeg/png) en la carpeta.
- Imágenes/s es EQUIVALENTE (1000 / mediana de ms_total), no throughput real.
- GPU %, VRAM y potencia son del DISPOSITIVO (NVML); `VRAM pico torch` es del proceso.
- `--comparar-precision` corre fp16 y fp32 en CUDA reconstruyendo el pipeline.

Autor: Dev-benchmark (RoadVision)
Fecha: 2026-11-06
"""
from __future__ import annotations

import argparse
import dataclasses
import gc
import random
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image  # noqa: E402

from roadvision.config import Config  # noqa: E402

if TYPE_CHECKING:
    from rich.console import Console

    from roadvision.metrics import MetricasAnalisis
    from roadvision.pipeline import Pipeline

EXTENSIONES = {".jpg", ".jpeg", ".png"}
ETAPAS = ("total", "preproceso", "deteccion", "clasificacion", "respaldo")
MIN_N_P95 = 20
SEMILLA = 1234
TAM_SINTETICA = (1280, 720)
N_SINTETICAS = 3


# --- Imágenes ------------------------------------------------------------------------------
def imagenes_sinteticas(n: int = N_SINTETICAS) -> list[Image.Image]:
    """Imágenes de ruido determinista (semilla fija) de 1280x720."""
    ancho, alto = TAM_SINTETICA
    imagenes = []
    for i in range(n):
        datos = random.Random(SEMILLA + i).randbytes(ancho * alto * 3)
        imagenes.append(Image.frombytes("RGB", (ancho, alto), datos))
    return imagenes


def cargar_imagenes(carpeta: Path) -> tuple[list[Image.Image], bool]:
    """(imágenes, son_sinteticas). Ordenadas por nombre; sintéticas si no hay fotos."""
    rutas: list[Path] = []
    if carpeta.is_dir():
        rutas = sorted(
            (p for p in carpeta.iterdir() if p.suffix.lower() in EXTENSIONES),
            key=lambda p: p.name.lower())
    imagenes: list[Image.Image] = []
    for ruta in rutas:
        try:
            with Image.open(ruta) as f:
                imagenes.append(f.convert("RGB"))
        except (OSError, Image.DecompressionBombError):
            print(f"Aviso: se omite {ruta.name} (no se pudo abrir).", file=sys.stderr)
    if imagenes:
        return imagenes, False
    return imagenes_sinteticas(), True


# --- Núcleo testeable ----------------------------------------------------------------------
def correr(
    pipeline: Pipeline,
    imagenes: Sequence[Image.Image],
    repeticiones: int,
    *,
    ruta_csv: Path | None = None,
) -> list[MetricasAnalisis]:
    """Analiza cada imagen `repeticiones` veces (SIN calentar: eso lo hace el llamador).

    Registra cada fila en `ruta_csv` con `escribir_csv` si no es None.
    """
    from roadvision.metrics import analizar_medido, escribir_csv

    filas: list[MetricasAnalisis] = []
    for _ in range(repeticiones):
        for imagen in imagenes:
            _, metricas = analizar_medido(pipeline, imagen)
            filas.append(metricas)
            if ruta_csv is not None:
                escribir_csv(metricas, ruta_csv)
    return filas


def _percentil(ordenados: list[float], p: float) -> float:
    """Percentil con interpolación lineal sobre una lista ya ordenada."""
    if len(ordenados) == 1:
        return ordenados[0]
    pos = (len(ordenados) - 1) * p / 100
    bajo = int(pos)
    alto = min(bajo + 1, len(ordenados) - 1)
    return ordenados[bajo] + (ordenados[alto] - ordenados[bajo]) * (pos - bajo)


def _estadisticas(valores: list[float]) -> dict[str, Any] | None:
    if not valores:
        return None
    v = sorted(valores)
    return {
        "n": len(v),
        "media": sum(v) / len(v),
        "p50": _percentil(v, 50),
        "p95": _percentil(v, 95) if len(v) >= MIN_N_P95 else None,
        "min": v[0],
        "max": v[-1],
    }


def _max(valores: list[float]) -> float | None:
    return max(valores) if valores else None


def _prom(valores: list[float]) -> float | None:
    return sum(valores) / len(valores) if valores else None


def _col(filas: list[MetricasAnalisis], campo: str) -> list[float]:
    return [getattr(f, campo) for f in filas if getattr(f, campo) is not None]


def resumir(filas: list[MetricasAnalisis]) -> dict[str, Any]:
    """Resumen PURO de las filas medidas. Ignora filas de calentamiento.

    Claves: n, dispositivo, precision, etapas {etapa: estadísticas|None},
    imagenes_por_s_equivalente, gpu_pct_pico, vram_usada_mb_pico (dispositivo),
    vram_pico_torch_mb, temperatura_c_max, potencia_w_prom, cpu_pct_prom,
    ram_proceso_mb_pico, detecciones_total.

    Lanza ValueError si las filas medidas mezclan dispositivo o precisión.
    """
    filas = [f for f in filas if not f.calentamiento]
    configs = {(f.dispositivo, f.precision) for f in filas}
    if len(configs) > 1:
        raise ValueError(f"resumir recibió configuraciones mezcladas: {sorted(configs)}")
    etapas = {e: _estadisticas(_col(filas, f"ms_{e}")) for e in ETAPAS}
    total = etapas["total"]
    return {
        "n": len(filas),
        "dispositivo": filas[0].dispositivo if filas else None,
        "precision": filas[0].precision if filas else None,
        "etapas": etapas,
        "imagenes_por_s_equivalente": 1000 / total["p50"] if total and total["p50"] else None,
        "gpu_pct_pico": _max(_col(filas, "gpu_pct_pico")),
        "vram_usada_mb_pico": _max(_col(filas, "vram_usada_mb")),
        "vram_pico_torch_mb": _max(_col(filas, "vram_pico_torch_mb")),
        "temperatura_c_max": _max(_col(filas, "temperatura_c_max")),
        "potencia_w_prom": _prom(_col(filas, "potencia_w_prom")),
        "cpu_pct_prom": _prom(_col(filas, "cpu_pct_prom")),
        "ram_proceso_mb_pico": _max(_col(filas, "ram_proceso_mb_pico")),
        "detecciones_total": sum(f.detecciones for f in filas),
    }


# --- Presentación --------------------------------------------------------------------------
def _f(valor: float | None, unidad: str = "", dec: int = 1) -> str:
    return "n/d" if valor is None else f"{valor:.{dec}f}{unidad}"


def mostrar_resumen(
    resumen: dict[str, Any], consola: Console | None = None, sinteticas: bool = False
) -> None:
    from rich.console import Console
    from rich.table import Table

    consola = consola or Console()
    titulo = (f"Tiempos por etapa (ms) - {resumen['dispositivo']} / "
              f"{resumen['precision']} - sin calentamiento")
    if sinteticas:
        titulo += " - IMÁGENES SINTÉTICAS (no representativas)"
    t = Table(title=titulo)
    for c in ("Etapa", "n", "media", "p50", "p95", "mín", "máx"):
        t.add_column(c, justify="left" if c == "Etapa" else "right")
    for etapa, e in resumen["etapas"].items():
        if e is None:
            t.add_row(f"ms_{etapa}", "0", *["n/d"] * 5)
            continue
        p95 = _f(e["p95"]) if e["p95"] is not None else f"n/a (n<{MIN_N_P95})"
        rotulo = f"ms_{etapa}"
        if etapa in ("clasificacion", "respaldo"):
            rotulo += "*"
        t.add_row(rotulo, str(e["n"]), _f(e["media"]), _f(e["p50"]), p95,
                  _f(e["min"]), _f(e["max"]))
    consola.print(t)
    consola.print("* costo por imagen; incluye 0.0 ms si no hubo recortes.")

    en_cpu = resumen["dispositivo"] == "CPU"
    na = "n/a (corrida en CPU)"
    r = Table(title="Consumo")
    r.add_column("Métrica")
    r.add_column("Valor", justify="right")
    filas = [
        ("Imágenes/s (equivalente, 1000/mediana ms_total)",
         _f(resumen["imagenes_por_s_equivalente"], "", 2)),
        ("GPU % pico (dispositivo)", na if en_cpu else _f(resumen["gpu_pct_pico"], " %", 0)),
        ("VRAM pico (dispositivo)",
         na if en_cpu else _f(resumen["vram_usada_mb_pico"], " MB", 0)),
        ("VRAM pico torch (proceso)", _f(resumen["vram_pico_torch_mb"], " MB", 0)),
        ("Temperatura máx (dispositivo)",
         na if en_cpu else _f(resumen["temperatura_c_max"], " °C", 0)),
        ("Potencia prom (dispositivo)",
         na if en_cpu else _f(resumen["potencia_w_prom"], " W")),
        ("CPU % prom (proceso, 100 % = 1 núcleo)", _f(resumen["cpu_pct_prom"], " %")),
        ("RAM pico (proceso)", _f(resumen["ram_proceso_mb_pico"], " MB", 0)),
    ]
    for nombre, valor in filas:
        r.add_row(nombre, valor)
    consola.print(r)


def mostrar_comparacion(
    resumenes: dict[str, dict[str, Any] | str], consola: Console | None = None
) -> None:
    """Tabla lado a lado. Cada valor es un resumen o un texto de error (p. ej. OOM)."""
    from rich.console import Console
    from rich.table import Table

    consola = consola or Console()
    t = Table(title="Comparación de precisión (mismo dispositivo)")
    t.add_column("Métrica")
    for nombre in resumenes:
        t.add_column(nombre, justify="right")

    def fila(titulo: str, extraer: Callable[[dict[str, Any]], str]) -> None:
        t.add_row(titulo, *[
            extraer(r) if isinstance(r, dict) else "fallo" for r in resumenes.values()])

    fila("Mediana ms_total", lambda r: _f(r["etapas"]["total"]["p50"] if r["etapas"]["total"]
                                          else None, " ms"))
    fila("Detecciones totales", lambda r: str(r["detecciones_total"]))
    fila("VRAM pico dispositivo", lambda r: _f(r["vram_usada_mb_pico"], " MB", 0))
    fila("VRAM pico torch", lambda r: _f(r["vram_pico_torch_mb"], " MB", 0))
    fila("Potencia prom", lambda r: _f(r["potencia_w_prom"], " W"))
    consola.print(t)
    for nombre, r in resumenes.items():
        if isinstance(r, str):
            consola.print(f"[red]{nombre}: {r}[/red]")


# --- Ejecución -----------------------------------------------------------------------------
def _liberar() -> None:
    gc.collect()
    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()


def _construir_por_defecto(config: Config) -> Pipeline:
    from roadvision.pipeline import construir_pipeline  # import perezoso (arrastra torch)

    return construir_pipeline(config)


def _medir_config(
    config: Config,
    construir: Callable[[Config], Pipeline],
    imagenes: Sequence[Image.Image],
    repeticiones: int,
    ruta_csv: Path | None,
    consola: Console,
) -> dict[str, Any]:
    """Construye, calienta (aparte), corre y resume. Libera el pipeline al terminar."""
    from roadvision.metrics import calentar_medido, escribir_csv, reporte_arranque

    pipeline = construir(config)
    try:
        calentamiento = calentar_medido(pipeline)
        if ruta_csv is not None:
            escribir_csv(calentamiento, ruta_csv)
        reporte_arranque(pipeline, calentamiento, consola=consola)
        consola.print(
            f"Dispositivo y precisión en uso: {pipeline.dispositivo.descripcion} / "
            f"{calentamiento.precision}")
        consola.print(f"Midiendo: {len(imagenes)} imagen(es) x {repeticiones} repeticiones...")
        filas = correr(pipeline, imagenes, repeticiones, ruta_csv=ruta_csv)
        resumen = resumir(filas)
        resumen["_nombre_dispositivo"] = pipeline.dispositivo.nombre
        return resumen
    finally:
        del pipeline
        _liberar()


def main(
    argv: Sequence[str] | None = None,
    construir: Callable[[Config], Pipeline] | None = None,
) -> int:
    for flujo in (sys.stdout, sys.stderr):
        if hasattr(flujo, "reconfigure"):
            try:
                flujo.reconfigure(encoding="utf-8", errors="replace")
            except (OSError, ValueError):
                pass

    parser = argparse.ArgumentParser(
        prog="benchmark", description="Benchmark de RoadVision (tiempos, GPU y VRAM).")
    parser.add_argument("--carpeta", default="samples", help="carpeta con fotos (jpg/png)")
    parser.add_argument("--repeticiones", type=int, default=5)
    parser.add_argument("--dispositivo", choices=["auto", "cpu", "cuda"], default="auto")
    precision = parser.add_mutually_exclusive_group()
    precision.add_argument("--sin-fp16", action="store_true", help="desactiva media precisión")
    precision.add_argument("--comparar-precision", action="store_true",
                           help="corre fp16 y fp32 en el mismo dispositivo y los compara")
    destino = parser.add_mutually_exclusive_group()
    destino.add_argument("--csv", default=None, help="CSV de métricas (por defecto el del repo)")
    destino.add_argument("--sin-csv", action="store_true", help="no escribir CSV")
    parser.add_argument("--pausa-s", type=float, default=5.0,
                        help="pausa entre fp16 y fp32 para enfriar la GPU (0 la quita)")
    args = parser.parse_args(argv)

    if args.repeticiones < 1:
        print("--repeticiones debe ser al menos 1.", file=sys.stderr)
        return 2
    if args.comparar_precision and args.dispositivo == "cpu":
        print("--comparar-precision requiere CUDA (en CPU siempre es fp32).", file=sys.stderr)
        return 2

    carpeta = Path(args.carpeta)
    if not carpeta.is_dir():
        print(f"La carpeta '{carpeta}' no existe.", file=sys.stderr)
        return 2

    from rich.console import Console

    consola = Console()
    construir_por_defecto = construir is None
    construir = construir or _construir_por_defecto

    if args.sin_csv:
        ruta_csv = None
    elif args.csv:
        ruta_csv = Path(args.csv)
    else:
        from roadvision.metrics import RUTA_CSV

        ruta_csv = RUTA_CSV

    imagenes, sinteticas = cargar_imagenes(carpeta)
    if sinteticas and args.csv is None and not args.sin_csv:
        ruta_csv = None
        consola.print(
            "[yellow]Con imágenes sintéticas no se escribe el CSV del repo "
            "(use --csv RUTA para forzarlo).[/yellow]")
    if sinteticas:
        consola.print(
            f"[yellow]AVISO: no hay fotos en '{args.carpeta}'; se usan {len(imagenes)} imágenes "
            f"SINTÉTICAS de ruido ({TAM_SINTETICA[0]}x{TAM_SINTETICA[1]}). No habrá señales "
            "detectadas; para resultados reales ponga fotos reales (jpg/png) en la "
            "carpeta.[/yellow]")
    else:
        consola.print(f"{len(imagenes)} imagen(es) cargadas de '{args.carpeta}'.")

    base = dataclasses.replace(Config(), dispositivo=args.dispositivo, fp16=not args.sin_fp16)
    try:
        if args.comparar_precision:
            if construir_por_defecto:
                import torch

                if not torch.cuda.is_available():
                    print("--comparar-precision requiere CUDA y no hay una GPU disponible.",
                          file=sys.stderr)
                    return 2
            return _comparar(base, construir, imagenes, args.repeticiones, ruta_csv, consola,
                             args.pausa_s, sinteticas)
        resumen = _medir_config(base, construir, imagenes, args.repeticiones, ruta_csv, consola)
    except ImportError as exc:
        print(f"No se pudo cargar PyTorch o una dependencia ({exc}). "
              "Instale las dependencias del proyecto (requirements.txt).",
              file=sys.stderr)
        return 2
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    except OSError as exc:
        print(f"No se pudieron cargar los modelos (¿no descargados o sin red?): {exc}",
              file=sys.stderr)
        return 2

    mostrar_resumen(resumen, consola, sinteticas)
    if ruta_csv is not None:
        consola.print(f"Filas registradas en {ruta_csv} (el calentamiento va marcado).")
    return 0


def _comparar(
    base: Config,
    construir: Callable[[Config], Pipeline],
    imagenes: Sequence[Image.Image],
    repeticiones: int,
    ruta_csv: Path | None,
    consola: Console,
    pausa_s: float = 0.0,
    sinteticas: bool = False,
) -> int:
    resumenes: dict[str, dict[str, Any] | str] = {}
    for nombre, fp16 in (("fp16", True), ("fp32", False)):
        if nombre == "fp32" and pausa_s > 0:
            consola.print(f"Enfriando {pausa_s:g} s antes de fp32 (evita inflarlo por "
                          "throttling térmico)...")
            time.sleep(pausa_s)
        consola.print(f"[bold]== Precisión {nombre} ==[/bold]")
        try:
            resumen = _medir_config(
                dataclasses.replace(base, fp16=fp16), construir, imagenes, repeticiones,
                ruta_csv, consola)
        except RuntimeError as exc:
            # CUDA sin disponibilidad o torch.cuda.OutOfMemoryError (subclase de RuntimeError).
            if not resumenes:  # falló ya el primero: sin CUDA, no hay comparación posible
                print(f"Error: {exc}", file=sys.stderr)
                return 2
            resumenes[nombre] = f"no se pudo completar ({type(exc).__name__}: {exc})"
            _liberar()
            continue
        if resumen["_nombre_dispositivo"] != "cuda":
            print("La comparación fp16/fp32 requiere CUDA; el dispositivo en uso es CPU "
                  "(siempre fp32).", file=sys.stderr)
            return 2
        mostrar_resumen(resumen, consola, sinteticas)
        resumenes[nombre] = resumen
    mostrar_comparacion(resumenes, consola)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
