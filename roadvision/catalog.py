"""Catálogo de señales colombianas: carga, validación y consultas."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import yaml

from roadvision.contracts import Categoria

ID_VELOCIDAD_NO_VERIFICADA = "velocidad_no_verificada"


class ErrorCatalogo(ValueError):
    """El archivo del catálogo es inválido."""


@dataclass(frozen=True)
class EntradaCatalogo:
    id: str
    nombre: str
    categoria: Categoria
    significado: str
    gtsrb: tuple[str, ...]
    clip: str | None
    velocidad_kmh: int | None


@dataclass(frozen=True)
class InfoCategoria:
    nombre: str
    significado: str
    clip: str


@dataclass(frozen=True)
class Catalogo:
    entradas: dict[str, EntradaCatalogo]
    categorias: dict[Categoria, InfoCategoria]
    clip_otro: str
    nombre_no_reconocida: str
    gtsrb_sin_equivalente: frozenset[str]

    def por_etiqueta_gtsrb(self, etiqueta: str) -> EntradaCatalogo | None:
        for entrada in self.entradas.values():
            if etiqueta in entrada.gtsrb:
                return entrada
        return None

    def velocidades(self) -> list[EntradaCatalogo]:
        return [e for e in self.entradas.values()
                if e.velocidad_kmh is not None and e.clip]

    def con_clip(self) -> list[EntradaCatalogo]:
        return [e for e in self.entradas.values() if e.clip]


def _requerido(datos: dict, clave: str, contexto: str):
    if clave not in datos or datos[clave] in (None, ""):
        raise ErrorCatalogo(f"Falta '{clave}' en {contexto}")
    return datos[clave]


def _categoria(valor: str, contexto: str) -> Categoria:
    try:
        return Categoria(valor)
    except ValueError:
        raise ErrorCatalogo(f"Categoría inválida '{valor}' en {contexto}") from None


def cargar_catalogo(ruta: Path) -> Catalogo:
    try:
        datos = yaml.safe_load(Path(ruta).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        raise ErrorCatalogo(f"No se pudo leer el catálogo: {exc}") from exc
    if not isinstance(datos, dict):
        raise ErrorCatalogo("El catálogo debe ser un mapa YAML")

    categorias: dict[Categoria, InfoCategoria] = {}
    crudas = _requerido(datos, "categorias", "el catálogo")
    if not isinstance(crudas, dict):
        raise ErrorCatalogo("'categorias' debe ser un mapa")
    for clave, info in crudas.items():
        cat = _categoria(clave, "'categorias'")
        if not isinstance(info, dict):
            raise ErrorCatalogo(f"La categoría '{clave}' debe ser un mapa")
        categorias[cat] = InfoCategoria(
            nombre=_requerido(info, "nombre", f"categoría {clave}"),
            significado=_requerido(info, "significado", f"categoría {clave}"),
            clip=_requerido(info, "clip", f"categoría {clave}"),
        )
    faltan = set(Categoria) - set(categorias)
    if faltan:
        raise ErrorCatalogo(f"Faltan categorías: {sorted(c.value for c in faltan)}")

    entradas: dict[str, EntradaCatalogo] = {}
    dueno: dict[str, str] = {}
    lista = _requerido(datos, "senales", "el catálogo")
    if not isinstance(lista, list):
        raise ErrorCatalogo("'senales' debe ser una lista")
    for crudo in lista:
        if not isinstance(crudo, dict):
            raise ErrorCatalogo("Cada elemento de 'senales' debe ser un mapa")
        id_ = _requerido(crudo, "id", "una señal")
        if id_ in entradas:
            raise ErrorCatalogo(f"Id repetido: {id_}")
        gtsrb = crudo.get("gtsrb") or []
        if not isinstance(gtsrb, list) or not all(isinstance(g, str) for g in gtsrb):
            raise ErrorCatalogo(f"'gtsrb' de '{id_}' debe ser una lista de textos")
        gtsrb = tuple(gtsrb)
        for etiqueta in gtsrb:
            if etiqueta in dueno:
                raise ErrorCatalogo(
                    f"Etiqueta GTSRB '{etiqueta}' en '{dueno[etiqueta]}' y '{id_}'")
            dueno[etiqueta] = id_
        entradas[id_] = EntradaCatalogo(
            id=id_,
            nombre=_requerido(crudo, "nombre", id_),
            categoria=_categoria(_requerido(crudo, "categoria", id_), id_),
            significado=_requerido(crudo, "significado", id_),
            gtsrb=gtsrb,
            clip=crudo.get("clip"),
            velocidad_kmh=crudo.get("velocidad_kmh"),
        )

    crudas_sin = datos.get("gtsrb_sin_equivalente") or []
    if not isinstance(crudas_sin, list) or not all(isinstance(g, str) for g in crudas_sin):
        raise ErrorCatalogo("'gtsrb_sin_equivalente' debe ser una lista de textos")
    sin_equivalente = frozenset(crudas_sin)
    solapadas = sin_equivalente & set(dueno)
    if solapadas:
        raise ErrorCatalogo(
            f"Etiquetas mapeadas y también sin equivalente: {sorted(solapadas)}")

    generica = entradas.get(ID_VELOCIDAD_NO_VERIFICADA)
    if generica is None:
        raise ErrorCatalogo(f"Falta la entrada '{ID_VELOCIDAD_NO_VERIFICADA}'")
    if generica.categoria is not Categoria.REGLAMENTARIA:
        raise ErrorCatalogo(f"'{ID_VELOCIDAD_NO_VERIFICADA}' debe ser reglamentaria")

    return Catalogo(
        entradas=entradas,
        categorias=categorias,
        clip_otro=_requerido(datos, "clip_otro", "el catálogo"),
        nombre_no_reconocida=_requerido(datos, "no_reconocida", "el catálogo"),
        gtsrb_sin_equivalente=sin_equivalente,
    )


def etiquetas_sin_cubrir(catalogo: Catalogo, etiquetas: Iterable[str]) -> list[str]:
    return [e for e in etiquetas
            if catalogo.por_etiqueta_gtsrb(e) is None
            and e not in catalogo.gtsrb_sin_equivalente]
