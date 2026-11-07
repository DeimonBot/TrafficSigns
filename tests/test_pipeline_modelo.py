from __future__ import annotations

import pytest

from roadvision.cajas import iou
from roadvision.pipeline import construir_pipeline

pytestmark = pytest.mark.modelos


def test_pipeline_real_reconoce_pare(escena_con_pare):
    p = construir_pipeline()
    img, caja = escena_con_pare(800, 600, 420, 120, 128)
    r = p.analizar(img)
    assert any(s.id_catalogo == "pare" and iou(s.caja, caja) >= 0.3 for s in r.senales)
    assert set(p.carga) == {"detector", "clasificador", "respaldo"}
