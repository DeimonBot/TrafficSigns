from __future__ import annotations

import time

from roadvision.timing import Cronometro


def test_cronometro_suma_y_sincroniza():
    llamadas = []
    c = Cronometro(sincronizar=lambda: llamadas.append(1))
    with c.etapa("a"):
        time.sleep(0.01)
    with c.etapa("a"):
        pass
    assert c.tiempos_ms["a"] >= 10
    assert len(llamadas) == 4


def test_cronometro_sin_sincronizar():
    c = Cronometro()
    with c.etapa("b"):
        pass
    assert c.tiempos_ms["b"] >= 0
