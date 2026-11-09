"""Regresión del embedding SigLIP en ejecución vs embeddings guardados (marca `modelos`)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from roadvision.capa_segura import EmbedderSiglip, UmbralesCapa, cargar_cabezas, cargar_meta, predecir
from tests.fakes_capa import catalogo_real

pytestmark = pytest.mark.modelos
FIXT = Path(__file__).parent / "fixtures" / "capa_segura"


def test_embedding_en_ejecucion_coincide_con_el_guardado(dispositivo):
    ref = np.load(FIXT / "referencia.npz", allow_pickle=False)
    faltan = [str(a) for a in ref["archivos"] if not (FIXT / str(a)).exists()]
    if faltan:
        # Los recortes PNG no se versionan (vienen de fotos con licencia CC y solo se usaron
        # para entrenar): sin ellos esta regresión se omite en vez de fallar.
        pytest.skip(f"faltan {len(faltan)} recortes de referencia en {FIXT}")
    meta = cargar_meta()
    emb = EmbedderSiglip(meta.encoder, dispositivo, meta.revision)
    imgs = [Image.open(FIXT / str(a)).convert("RGB") for a in ref["archivos"]]
    E = emb.embeber(imgs, tamano_lote=8)
    n = lambda x: x / np.linalg.norm(x, axis=1, keepdims=True)
    cos = (n(E) * n(ref["emb"].astype(np.float64))).sum(1)
    assert E.shape == ref["emb"].shape
    assert cos.min() >= 0.999, cos.min()
    # Y la decisión de la capa es la misma que con los embeddings guardados.
    _, cl, _, _, nb = predecir(E, cargar_cabezas(catalogo_real()), UmbralesCapa(*ref["umbrales"]))
    assert list(cl) == list(ref["clase"]) and list(nb) == list(ref["nombra"])
