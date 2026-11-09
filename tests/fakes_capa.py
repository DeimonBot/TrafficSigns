"""Dobles para la capa segura: cabezas sintéticas y embedder falso (sin torch ni modelos)."""
from __future__ import annotations

import numpy as np
import torch
from PIL import Image

from roadvision.capa_segura import Cabezas, PipelineSeguro, UmbralesCapa
from roadvision.catalog import Catalogo, cargar_catalogo
from roadvision.config import Config
from roadvision.contracts import Deteccion, Prediccion
from roadvision.device import Dispositivo
from tests.fakes import FakeClasificador, FakeDetector, FakeRespaldo

CLASES = ("curva_izquierda", "otra_senal", "pare", "resalto", "velocidad_50")
K = len(CLASES)


def catalogo_real() -> Catalogo:
    return cargar_catalogo(Config().ruta_catalogo)


def cabezas_falsas() -> Cabezas:
    """Dim 0 = puerta; dims 1..K = una por clase. logit puerta = 20*e0 - 10; logit clase = 12*e_k."""
    d = K + 1
    W = np.zeros((d, K))
    for j in range(K):
        W[1 + j, j] = 12.0
    gw = np.zeros(d)
    gw[0] = 20.0
    return Cabezas(
        gate_mu=np.zeros(d), gate_sd=np.ones(d), gate_w=gw, gate_b=-10.0,
        id_mu=np.zeros(d), id_sd=np.ones(d), id_W=W, id_b=np.zeros(K), T=1.0, clases=CLASES)


def emb(puerta: float, **clases: float) -> np.ndarray:
    """Vector crudo (se normaliza en `predecir`): componente de puerta y de cada clase."""
    v = np.zeros(K + 1)
    v[0] = puerta
    for nombre, valor in clases.items():
        v[1 + CLASES.index(nombre)] = valor
    return v


# Vectores con puerta/identidad conocidas (una prueba verifica las precondiciones).
def e_nombra(clase: str) -> np.ndarray:         # puerta ~0.98, identidad ~1.0
    return emb(1.0, **{clase: 1.0})


def e_puerta_093(clase: str) -> np.ndarray:     # puerta ~0.93
    return emb(1.0, **{clase: 1.233})


def e_puerta_060(clase: str) -> np.ndarray:     # puerta ~0.60
    return emb(1.0, **{clase: 1.641})


def e_puerta_baja(clase: str) -> np.ndarray:    # puerta ~0.002
    return emb(0.2, **{clase: 1.0})


class FakeEmbedder:
    """Embedder falso: el embedding de cada recorte se decide por su ancho en píxeles."""

    def __init__(self, por_ancho: dict[int, np.ndarray]) -> None:
        self.por_ancho = por_ancho
        self.llamadas: list[int] = []

    def embeber(self, recortes: list[Image.Image], tamano_lote: int = 32) -> np.ndarray:
        self.llamadas.append(len(recortes))
        if not recortes:
            return np.zeros((0, K + 1))
        return np.stack([self.por_ancho.get(r.width, e_puerta_baja("pare")) for r in recortes])


def ancho_recorte(ancho_caja: int) -> int:
    """Ancho del recorte de una caja cuadrada con margen 10 % por lado."""
    return round(ancho_caja * 1.2)


def pipeline_seguro_falso(
    casos: list[tuple[int, np.ndarray, Prediccion]],
    respaldo: FakeRespaldo | None = None,
    umbrales_capa: UmbralesCapa | None = None,
) -> PipelineSeguro:
    """`casos`: (ancho de caja, embedding, predicción ViT); cajas separadas en x."""
    detecciones = [Deteccion((20 + i * 200, 50, 20 + i * 200 + a, 50 + a), 0.9)
                   for i, (a, _, _) in enumerate(casos)]
    cat = catalogo_real()
    return PipelineSeguro(
        Config(), FakeDetector(detecciones),
        FakeClasificador([p for _, _, p in casos] or [Prediccion("Stop", 0.9)]),
        respaldo or FakeRespaldo(por_defecto={cat.clip_otro: 5}), cat,
        Dispositivo("cpu", torch.float32, "CPU"),
        FakeEmbedder({ancho_recorte(a): e for a, e, _ in casos}), cabezas_falsas(),
        umbrales_capa or UmbralesCapa())
