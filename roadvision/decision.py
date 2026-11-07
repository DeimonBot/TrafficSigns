"""Reglas de decisión: cuándo se muestra un nombre de señal y cuándo no."""
from __future__ import annotations

from dataclasses import dataclass

from PIL import Image

from roadvision.catalog import Catalogo, EntradaCatalogo
from roadvision.config import Umbrales
from roadvision.contracts import Categoria, Estado, Fuente, Prediccion, Respaldo

ID_VELOCIDAD_NO_VERIFICADA = "velocidad_no_verificada"
_ORDEN_CATEGORIAS = (Categoria.REGLAMENTARIA, Categoria.PREVENTIVA, Categoria.INFORMATIVA)


@dataclass(frozen=True)
class Decision:
    estado: Estado
    fuente: Fuente
    id_catalogo: str | None
    categoria: Categoria | None
    confianza: float


_NO_RECONOCIDA = Decision(Estado.NO_RECONOCIDA, Fuente.NINGUNA, None, None, 0.0)


def _mejor(fila: list[float]) -> tuple[int, float]:
    indice = max(range(len(fila)), key=fila.__getitem__)
    return indice, fila[indice]


class Decisor:
    def __init__(self, catalogo: Catalogo, respaldo: Respaldo) -> None:
        self._catalogo = catalogo
        self._respaldo = respaldo

    def decidir(
        self,
        recortes: list[Image.Image],
        predicciones: list[Prediccion],
        umbrales: Umbrales,
    ) -> list[Decision]:
        cat = self._catalogo
        resultado: list[Decision | None] = [None] * len(recortes)
        verificar: list[tuple[int, EntradaCatalogo]] = []  # (índice, entrada de velocidad)
        respaldo: list[int] = []

        for i, pred in enumerate(predicciones):
            entrada = cat.por_etiqueta_gtsrb(pred.etiqueta)
            if (
                pred.etiqueta in cat.gtsrb_sin_equivalente
                or pred.probabilidad < umbrales.gtsrb
                or entrada is None
            ):
                respaldo.append(i)
            elif entrada.velocidad_kmh is not None:
                verificar.append((i, entrada))
            else:
                resultado[i] = Decision(
                    Estado.RECONOCIDA, Fuente.GTSRB, entrada.id, entrada.categoria,
                    pred.probabilidad)

        if verificar:
            velocidades = cat.velocidades()
            filas = self._respaldo.puntuar(
                [recortes[i] for i, _ in verificar], [v.clip for v in velocidades])
            for (i, entrada), fila in zip(verificar, filas):
                k, p = _mejor(fila)
                mejor = velocidades[k]
                prob = predicciones[i].probabilidad
                if mejor.velocidad_kmh == entrada.velocidad_kmh:
                    resultado[i] = Decision(
                        Estado.RECONOCIDA, Fuente.GTSRB, entrada.id, entrada.categoria, prob)
                elif mejor.gtsrb == () and p >= umbrales.clip_senal:
                    resultado[i] = Decision(
                        Estado.APROXIMADA, Fuente.CLIP, mejor.id, mejor.categoria, p)
                else:
                    resultado[i] = Decision(
                        Estado.SOLO_CATEGORIA, Fuente.GTSRB, ID_VELOCIDAD_NO_VERIFICADA,
                        Categoria.REGLAMENTARIA, prob)

        if respaldo:
            self._respaldar(recortes, respaldo, umbrales, resultado)

        return resultado  # type: ignore[return-value]

    def _respaldar(
        self,
        recortes: list[Image.Image],
        indices: list[int],
        umbrales: Umbrales,
        resultado: list[Decision | None],
    ) -> None:
        cat = self._catalogo
        senales = cat.con_clip()
        filas = self._respaldo.puntuar(
        	[recortes[i] for i in indices], [s.clip for s in senales])
        pendientes: list[int] = []
        for i, fila in zip(indices, filas):
            k, p = _mejor(fila)
            if p >= umbrales.clip_senal:
                s = senales[k]
                resultado[i] = Decision(Estado.APROXIMADA, Fuente.CLIP, s.id, s.categoria, p)
            else:
                pendientes.append(i)
        if not pendientes:
            return

        textos = [cat.categorias[c].clip for c in _ORDEN_CATEGORIAS] + [cat.clip_otro]
        filas = self._respaldo.puntuar([recortes[i] for i in pendientes], textos)
        for i, fila in zip(pendientes, filas):
            k, p = _mejor(fila)
            if k < len(_ORDEN_CATEGORIAS) and p >= umbrales.clip_categoria:
                resultado[i] = Decision(
                    Estado.SOLO_CATEGORIA, Fuente.CLIP, None, _ORDEN_CATEGORIAS[k], p)
            else:
                resultado[i] = _NO_RECONOCIDA
