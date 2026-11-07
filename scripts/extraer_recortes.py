"""Extrae con OWLv2 los recortes de señales de TUS fotos para etiquetarlos después.

Carga SOLO el detector OWLv2 (no el clasificador ni CLIP: ahorra VRAM), detecta con el umbral
de `Config.umbrales.deteccion`, filtra cajas, recorta con `Config.margen_recorte` y guarda
cada recorte como `{foto}__{n:02d}.png` más una fila en `indice.csv`.

Uso:
    python scripts/extraer_recortes.py [--fotos CARPETA] [--salida CARPETA]
        [--dispositivo auto|cpu|cuda] [--sin-fp16]

Por defecto: fotos en `datasets/co/fotos` y salida en `datasets/co` (recortes/, indice.csv).
Es idempotente: una foto ya indexada (mismo sha1) no se vuelve a procesar y un recorte que
ya existe en disco no se reescribe. Las fotos con contenido repetido se omiten.
Una foto sin detecciones no deja rastro en el índice, así que se reprocesa en la siguiente
corrida. Después, `streamlit run scripts/etiquetador.py`.

Códigos de salida: 0 ok; 2 uso/entrada incorrectos; 3 modelo no disponible.

Autor: jhoan
Fecha: 2026-11-06
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_RAIZ = Path(__file__).resolve().parents[1]
if str(_RAIZ) not in sys.path:
    sys.path.insert(0, str(_RAIZ))

from roadvision.config import Config  # noqa: E402
from roadvision.etiquetado import (  # noqa: E402
    Recorte,
    agregar_al_indice,
    fotos_indexadas,
    leer_indice,
    listar_fotos,
    nombre_recorte,
    rutas_por_defecto,
    sha1_archivo,
    stem_unico,
)


def _args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extrae recortes de señales con OWLv2.")
    p.add_argument("--fotos", type=Path, default=None, help="carpeta de fotos (jpg/jpeg/png)")
    p.add_argument("--salida", type=Path, default=None,
                   help="carpeta de salida (recortes/ e indice.csv); por defecto datasets/co")
    p.add_argument("--dispositivo", choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument("--sin-fp16", action="store_true", help="usar fp32 en CUDA")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    for flujo in (sys.stdout, sys.stderr):
        try:
            flujo.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass
    args = _args(argv)
    rutas = rutas_por_defecto(args.salida)
    carpeta_fotos = args.fotos if args.fotos is not None else rutas_por_defecto().fotos
    fotos = listar_fotos(carpeta_fotos)
    if not fotos:
        print(
            f"No hay fotos en '{carpeta_fotos}'.\n"
            "Copia tus fotos (jpg, jpeg o png) a esa carpeta, o indica otra con --fotos, y vuelve "
            "a correr:\n    python scripts/extraer_recortes.py", file=sys.stderr)
        return 2

    indice = leer_indice(rutas.indice)
    conocidas = fotos_indexadas(indice)  # sha1 -> nombre de foto
    hashes = {f: sha1_archivo(f) for f in fotos}
    ya_indexadas = sum(1 for f in fotos if conocidas.get(hashes[f]) == f.name)
    duplicadas = 0
    vistas: dict[str, str] = {}
    por_procesar: list[Path] = []
    for f in fotos:
        h = hashes[f]
        if h in conocidas:
            if conocidas[h] != f.name:
                duplicadas += 1
            continue
        if h in vistas:
            duplicadas += 1
            continue
        vistas[h] = f.name
        por_procesar.append(f)

    recortes_nuevos = 0
    if por_procesar:
        try:
            detector, config, dispositivo = _cargar_detector(args)
        except OSError as exc:
            print(
                "No se pudo cargar el modelo OWLv2 (¿no está descargado y no hay conexión?).\n"
                f"Detalle: {exc}\nDescárgalo con: python scripts/download_models.py",
                file=sys.stderr)
            return 3
        except (RuntimeError, ValueError) as exc:
            print(f"Error de dispositivo/modelo: {exc}", file=sys.stderr)
            return 2
        print(f"Dispositivo: {dispositivo.descripcion}. Procesando {len(por_procesar)} foto(s)...")
        from PIL import Image, UnidentifiedImageError

        from roadvision.cajas import filtrar_cajas
        from roadvision.imaging import normalizar, recortar

        rutas.recortes.mkdir(parents=True, exist_ok=True)
        for f in por_procesar:
            try:
                with Image.open(f) as crudo:
                    img = normalizar(crudo)
            except (UnidentifiedImageError, OSError) as exc:
                print(f"  ! {f.name}: no se pudo abrir ({exc}); se omite", file=sys.stderr)
                continue
            u = config.umbrales.deteccion
            dets = filtrar_cajas(
                detector.detectar(img, u), u, config.tamano_minimo_caja, config.iou_duplicados)
            dets.sort(key=lambda d: d.caja[:2])
            h = hashes[f]
            stem = stem_unico(f.stem, h, indice)
            filas: list[Recorte] = []
            for n, d in enumerate(dets, start=1):
                nombre = nombre_recorte(stem, n)
                destino = rutas.recortes / nombre
                recorte = recortar(img, d.caja, config.margen_recorte)
                if not destino.exists():
                    recorte.save(destino, format="PNG")
                filas.append(Recorte(nombre, f.name, d.caja, d.puntaje,
                                     recorte.width, recorte.height, h))
            escritos = agregar_al_indice(rutas.indice, filas)
            indice.extend(filas)
            recortes_nuevos += escritos
            print(f"  {f.name}: {len(dets)} recorte(s)")

    print(
        f"\nResumen: {len(fotos)} foto(s) en la carpeta, {len(por_procesar)} procesada(s), "
        f"{recortes_nuevos} recorte(s) nuevo(s), {ya_indexadas} ya indexada(s), "
        f"{duplicadas} duplicada(s) omitida(s).\n"
        f"Índice: {rutas.indice}\nSiguiente paso: streamlit run scripts/etiquetador.py")
    return 0


def _cargar_detector(args: argparse.Namespace):
    from roadvision.detector import DetectorOwlv2
    from roadvision.device import elegir_dispositivo

    config = Config()
    dispositivo = elegir_dispositivo(args.dispositivo, config.fp16 and not args.sin_fp16)
    detector = DetectorOwlv2(config.modelo_detector, config.consultas_detector, dispositivo)
    return detector, config, dispositivo


if __name__ == "__main__":
    raise SystemExit(main())
