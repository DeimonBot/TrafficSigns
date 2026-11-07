"""Lógica pura del etiquetado de recortes (sin torch ni streamlit).

Maneja el índice de recortes (`indice.csv`), las etiquetas del usuario (`etiquetas.csv`) y la
lista de clases para el clasificador (catálogo + 'otra_senal' + 'no_senal' + 'omitida').

Disposición por defecto: `ruta_datasets() / "co"` con `fotos/`, `recortes/`, `indice.csv` y
`etiquetas.csv`. Todo queda en `datasets/` (ignorado por git).

Escritura de archivos:
- `indice.csv` y `etiquetas.csv` crecen por append de una sola escritura por fila (con
  cabecera si el archivo es nuevo); un corte a mitad no deja filas partidas.
- `deshacer_ultima` reescribe `etiquetas.csv` de forma atómica (archivo temporal + os.replace).
- Una etiqueta posterior para el mismo recorte reemplaza a la anterior al leer (gana la última).

Autor: jhoan
Fecha: 2026-11-06
"""
from __future__ import annotations

import csv
import hashlib
import io
import os
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from roadvision.catalog import ID_VELOCIDAD_NO_VERIFICADA, Catalogo
from roadvision.config import ruta_datasets

COLUMNAS_INDICE = (
    "recorte", "foto", "caja_x1", "caja_y1", "caja_x2", "caja_y2",
    "puntaje", "ancho", "alto", "sha1_foto",
)
COLUMNAS_ETIQUETAS = ("recorte", "foto", "clase", "marca_tiempo")

CLASE_OTRA = "otra_senal"
CLASE_NO_SENAL = "no_senal"
CLASE_OMITIDA = "omitida"  # lo que guarda el botón "omitir"
META_MINIMA = 30  # ejemplos mínimos por clase para entrenar
EXTENSIONES_FOTO = (".jpg", ".jpeg", ".png")
GRUPOS = ("reglamentaria", "preventiva", "informativa", "especiales")


@dataclass(frozen=True)
class Recorte:
    recorte: str
    foto: str
    caja: tuple[int, int, int, int]
    puntaje: float
    ancho: int
    alto: int
    sha1_foto: str


@dataclass(frozen=True)
class Etiqueta:
    recorte: str
    foto: str
    clase: str
    marca_tiempo: str


@dataclass(frozen=True)
class Clase:
    id: str
    nombre: str
    grupo: str  # una de GRUPOS


@dataclass(frozen=True)
class Rutas:
    base: Path

    @property
    def fotos(self) -> Path:
        return self.base / "fotos"

    @property
    def recortes(self) -> Path:
        return self.base / "recortes"

    @property
    def indice(self) -> Path:
        return self.base / "indice.csv"

    @property
    def etiquetas(self) -> Path:
        return self.base / "etiquetas.csv"


def rutas_por_defecto(base: Path | None = None) -> Rutas:
    """Rutas del conjunto colombiano; `base` por defecto es `ruta_datasets() / "co"`."""
    return Rutas(Path(base) if base is not None else ruta_datasets() / "co")


# --------------------------------------------------------------------- utilidades

def sha1_archivo(ruta: Path) -> str:
    h = hashlib.sha1()
    with Path(ruta).open("rb") as f:
        for bloque in iter(lambda: f.read(1 << 20), b""):
            h.update(bloque)
    return h.hexdigest()


def listar_fotos(carpeta: Path) -> list[Path]:
    """jpg/jpeg/png (cualquier mayúscula) de la carpeta, ordenadas por nombre."""
    carpeta = Path(carpeta)
    if not carpeta.is_dir():
        return []
    return sorted(
        (p for p in carpeta.iterdir() if p.is_file() and p.suffix.lower() in EXTENSIONES_FOTO),
        key=lambda p: p.name.lower(),
    )


def nombre_recorte(stem: str, n: int) -> str:
    return f"{stem}__{n:02d}.png"


def stem_unico(stem: str, sha1: str, indice: Iterable[Recorte]) -> str:
    """`stem`, o `stem-<sha1[:8]>` si otro archivo distinto ya usa ese prefijo de recorte."""
    prefijo = f"{stem}__"
    for r in indice:
        if r.recorte.startswith(prefijo) and r.sha1_foto != sha1:
            return f"{stem}-{sha1[:8]}"
    return stem


def _leer_csv(ruta: Path, columnas: Sequence[str]) -> list[dict[str, str]]:
    ruta = Path(ruta)
    if not ruta.is_file():
        return []
    with ruta.open("r", encoding="utf-8", newline="") as f:
        lector = csv.DictReader(f)
        if lector.fieldnames is None or any(c not in lector.fieldnames for c in columnas):
            return []
        return [fila for fila in lector if fila.get(columnas[0])]


def _agregar_filas(ruta: Path, columnas: Sequence[str], filas: Iterable[Sequence[object]]) -> None:
    """Append: una sola escritura por llamada; cabecera solo si el archivo es nuevo/vacío."""
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    nuevo = not ruta.is_file() or ruta.stat().st_size == 0
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    if nuevo:
        w.writerow(columnas)
    for fila in filas:
        w.writerow(fila)
    with ruta.open("a", encoding="utf-8", newline="") as f:
        f.write(buf.getvalue())
        f.flush()
        os.fsync(f.fileno())


def _reescribir_atomico(ruta: Path, columnas: Sequence[str], filas: Iterable[Sequence[object]]) -> None:
    ruta = Path(ruta)
    tmp = ruta.with_name(ruta.name + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(columnas)
        w.writerows(filas)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, ruta)


# ------------------------------------------------------------------------ índice

def leer_indice(ruta: Path) -> list[Recorte]:
    salida: list[Recorte] = []
    for f in _leer_csv(ruta, COLUMNAS_INDICE):
        try:
            salida.append(Recorte(
                recorte=f["recorte"], foto=f["foto"],
                caja=(int(f["caja_x1"]), int(f["caja_y1"]), int(f["caja_x2"]), int(f["caja_y2"])),
                puntaje=float(f["puntaje"]), ancho=int(f["ancho"]), alto=int(f["alto"]),
                sha1_foto=f["sha1_foto"]))
        except (ValueError, KeyError):
            continue  # fila dañada: se ignora
    return salida


def agregar_al_indice(ruta: Path, nuevos: Iterable[Recorte]) -> int:
    """Agrega recortes que no estén ya (por nombre). Devuelve cuántos escribió."""
    existentes = {r.recorte for r in leer_indice(ruta)}
    filas = []
    for r in nuevos:
        if r.recorte in existentes:
            continue
        existentes.add(r.recorte)
        filas.append((r.recorte, r.foto, *r.caja, f"{r.puntaje:.4f}", r.ancho, r.alto, r.sha1_foto))
    if filas:
        _agregar_filas(ruta, COLUMNAS_INDICE, filas)
    return len(filas)


def fotos_indexadas(indice: Iterable[Recorte]) -> dict[str, str]:
    """sha1 -> nombre de la primera foto indexada con ese contenido."""
    mapa: dict[str, str] = {}
    for r in indice:
        mapa.setdefault(r.sha1_foto, r.foto)
    return mapa


def duplicadas_por_sha1(fotos: Iterable[tuple[str, str]]) -> dict[str, list[str]]:
    """De pares (nombre, sha1) devuelve sha1 -> nombres, solo los que se repiten."""
    grupos: dict[str, list[str]] = {}
    for nombre, sha1 in fotos:
        grupos.setdefault(sha1, []).append(nombre)
    return {k: v for k, v in grupos.items() if len(v) > 1}


# --------------------------------------------------------------------- etiquetas

def marca_tiempo_ahora() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def leer_etiquetas(ruta: Path) -> list[Etiqueta]:
    """Filas del archivo en orden de escritura (sin resolver repetidas)."""
    return [Etiqueta(f["recorte"], f["foto"], f["clase"], f["marca_tiempo"])
            for f in _leer_csv(ruta, COLUMNAS_ETIQUETAS)]


def etiquetas_vigentes(etiquetas: Iterable[Etiqueta]) -> dict[str, Etiqueta]:
    """Una etiqueta por recorte: gana la última escrita."""
    return {e.recorte: e for e in etiquetas}


def guardar_etiqueta(
    ruta: Path, recorte: str, foto: str, clase: str, marca_tiempo: str | None = None
) -> Etiqueta:
    e = Etiqueta(recorte, foto, clase, marca_tiempo or marca_tiempo_ahora())
    _agregar_filas(ruta, COLUMNAS_ETIQUETAS, [(e.recorte, e.foto, e.clase, e.marca_tiempo)])
    return e


def deshacer_ultima(ruta: Path) -> Etiqueta | None:
    """Borra la última fila escrita y la devuelve; None si no hay nada que deshacer."""
    etiquetas = leer_etiquetas(ruta)
    if not etiquetas:
        return None
    ultima = etiquetas[-1]
    _reescribir_atomico(
        ruta, COLUMNAS_ETIQUETAS,
        [(e.recorte, e.foto, e.clase, e.marca_tiempo) for e in etiquetas[:-1]])
    return ultima


def siguiente_sin_etiquetar(
    indice: Sequence[Recorte], etiquetas: Iterable[Etiqueta]
) -> Recorte | None:
    hechos = {e.recorte for e in etiquetas}
    for r in indice:
        if r.recorte not in hechos:
            return r
    return None


def conteos_por_clase(etiquetas: Iterable[Etiqueta]) -> dict[str, int]:
    """Ejemplos por clase para entrenar: usa la etiqueta vigente y excluye 'omitida'."""
    conteo: dict[str, int] = {}
    for e in etiquetas_vigentes(etiquetas).values():
        if e.clase == CLASE_OMITIDA:
            continue
        conteo[e.clase] = conteo.get(e.clase, 0) + 1
    return dict(sorted(conteo.items()))


def fotos_distintas(etiquetas: Iterable[Etiqueta]) -> int:
    """Cuántas fotos distintas aportan al menos un ejemplo de entrenamiento."""
    return len({e.foto for e in etiquetas_vigentes(etiquetas).values()
                if e.clase != CLASE_OMITIDA})


def clases_con_pocos(
    conteos: dict[str, int], clases: Iterable[str], minimo: int = META_MINIMA
) -> list[str]:
    """Clases (de `clases`) con menos de `minimo` ejemplos, incluidas las de 0."""
    return [c for c in clases if conteos.get(c, 0) < minimo]


# ----------------------------------------------------------------------- clases

def clases_disponibles(catalogo: Catalogo) -> dict[str, list[Clase]]:
    """Clases agrupadas: reglamentaria/preventiva/informativa (id y nombre del catálogo) y
    'especiales' (otra_senal, no_senal, omitir→'omitida'). Excluye la genérica de velocidad."""
    grupos: dict[str, list[Clase]] = {g: [] for g in GRUPOS}
    for entrada in catalogo.entradas.values():
        if entrada.id == ID_VELOCIDAD_NO_VERIFICADA:
            continue
        grupos[entrada.categoria.value].append(Clase(entrada.id, entrada.nombre, entrada.categoria.value))
    grupos["especiales"] = [
        Clase(CLASE_OTRA, "Otra señal (no está en el catálogo)", "especiales"),
        Clase(CLASE_NO_SENAL, "NO es una señal", "especiales"),
        Clase(CLASE_OMITIDA, "Omitir (no sé / mala calidad)", "especiales"),
    ]
    return grupos


def ids_de_entrenamiento(catalogo: Catalogo) -> list[str]:
    """Todas las clases que cuentan para entrenar (todas menos 'omitida')."""
    return [c.id for g in clases_disponibles(catalogo).values() for c in g
            if c.id != CLASE_OMITIDA]
