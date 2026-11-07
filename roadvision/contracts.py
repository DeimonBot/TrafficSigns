"""Tipos y protocolos compartidos por todo el núcleo de RoadVision."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol

from PIL import Image

# (x1, y1, x2, y2) en píxeles de la imagen normalizada.
Caja = tuple[int, int, int, int]


class Categoria(str, Enum):
    REGLAMENTARIA = "reglamentaria"
    PREVENTIVA = "preventiva"
    INFORMATIVA = "informativa"


class Fuente(str, Enum):
    GTSRB = "gtsrb"
    CLIP = "clip"
    NINGUNA = "ninguna"


class Estado(str, Enum):
    """RECONOCIDA: GTSRB + catálogo. APROXIMADA: mejor coincidencia de CLIP, se muestra como
    aproximada. SOLO_CATEGORIA: solo se conoce la categoría, o un límite de velocidad cuyo
    valor no se pudo verificar. NO_RECONOCIDA: ninguna de las anteriores."""

    RECONOCIDA = "reconocida"
    APROXIMADA = "aproximada"
    SOLO_CATEGORIA = "solo_categoria"
    NO_RECONOCIDA = "no_reconocida"


@dataclass(frozen=True)
class Deteccion:
    caja: Caja
    puntaje: float


@dataclass(frozen=True)
class Prediccion:
    etiqueta: str
    probabilidad: float


@dataclass
class Senal:
    numero: int
    caja: Caja
    recorte: Image.Image
    estado: Estado
    fuente: Fuente
    id_catalogo: str | None
    nombre: str
    categoria: Categoria | None
    significado: str | None
    confianza: float
    puntaje_deteccion: float


@dataclass
class Resultado:
    senales: list[Senal]
    conteo_por_nombre: dict[str, int]
    # Claves: `Categoria.value`, más "sin_categoria" para señales sin categoría.
    conteo_por_categoria: dict[str, int]
    # Siempre: total, preproceso, deteccion, clasificacion, respaldo.
    tiempos_ms: dict[str, float]
    ancho: int
    alto: int
    dispositivo: str


class Detector(Protocol):
    def detectar(self, imagen: Image.Image, umbral: float) -> list[Deteccion]: ...


class Clasificador(Protocol):
    etiquetas: list[str]  # ordenadas por id de clase

    def clasificar(self, recortes: list[Image.Image]) -> list[Prediccion]: ...


class Respaldo(Protocol):
    def puntuar(self, recortes: list[Image.Image], textos: list[str]) -> list[list[float]]:
        """Una fila por recorte: probabilidades (softmax) sobre `textos`."""
        ...
