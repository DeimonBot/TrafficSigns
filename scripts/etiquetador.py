"""Etiquetador de recortes de señales (Streamlit, en español, sin torch).

Ejecutar: `streamlit run scripts/etiquetador.py`

Muestra cada recorte detectado por OWLv2 (ver `scripts/extraer_recortes.py`) ampliado, y
cada botón de clase guarda la etiqueta en `etiquetas.csv` y pasa al siguiente recorte sin
etiquetar. Los falsos positivos (vallas, escudos de ruta, espejos) se etiquetan "NO es señal".
"Omitir" guarda la clase 'omitida': no vuelve a salir pero no cuenta para entrenar.

Las rutas salen de `ruta_datasets() / "co"` (variable ROADVISION_DATASETS para cambiarla).
Todo el estado vive en disco; `st.session_state` solo guarda mensajes y qué recorte se
muestra, por lo que reejecutar la app no la rompe.

Autor: jhoan
Fecha: 2026-11-06
"""
from __future__ import annotations

import sys
from pathlib import Path

# Permite `streamlit run` desde cualquier carpeta sin instalar el paquete.
_RAIZ = Path(__file__).resolve().parents[1]
if str(_RAIZ) not in sys.path:
    sys.path.insert(0, str(_RAIZ))

import streamlit as st  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

from roadvision.catalog import cargar_catalogo  # noqa: E402
from roadvision.config import Config  # noqa: E402
from roadvision.etiquetado import (  # noqa: E402
    CLASE_OMITIDA,
    META_MINIMA,
    clases_disponibles,
    clases_con_pocos,
    conteos_por_clase,
    deshacer_ultima,
    fotos_distintas,
    guardar_etiqueta,
    ids_de_entrenamiento,
    leer_etiquetas,
    leer_indice,
    rutas_por_defecto,
    siguiente_sin_etiquetar,
    etiquetas_vigentes,
)

TAM_RECORTE = 320  # lado mayor del recorte ampliado, en píxeles
ANCHO_FOTO = 900
COLUMNAS_BOTONES = 3
COLUMNAS_GALERIA = 6
TITULOS_GRUPO = {
    "reglamentaria": "Reglamentarias",
    "preventiva": "Preventivas",
    "informativa": "Informativas",
    "especiales": "Especiales",
}


@st.cache_resource(show_spinner=False)
def _catalogo():
    return cargar_catalogo(Config().ruta_catalogo)


def _ampliar(img: Image.Image) -> Image.Image:
    """Escala el lado mayor a TAM_RECORTE con interpolación bicúbica (suave pero legible)."""
    img = img.convert("RGB")
    factor = TAM_RECORTE / max(img.size)
    nuevo = (max(1, round(img.width * factor)), max(1, round(img.height * factor)))
    return img.resize(nuevo, Image.Resampling.BICUBIC)


def _foto_con_caja(ruta: Path, caja: tuple[int, int, int, int]) -> Image.Image | None:
    from roadvision.imaging import normalizar

    try:
        with Image.open(ruta) as crudo:
            img = normalizar(crudo)
    except (OSError, ValueError):
        return None
    x1, y1, x2, y2 = caja
    grosor = max(3, round(max(img.size) / 250))
    ImageDraw.Draw(img).rectangle((x1, y1, x2, y2), outline=(255, 0, 0), width=grosor)
    if img.width > ANCHO_FOTO:
        img = img.resize((ANCHO_FOTO, round(img.height * ANCHO_FOTO / img.width)),
                         Image.Resampling.LANCZOS)
    return img


def _al_etiquetar(rutas, recorte: str, foto: str, clase: str, nombre_clase: str) -> None:
    # Si el recorte ya quedó etiquetado (doble clic / pestaña vieja) no se duplica la fila.
    if recorte in etiquetas_vigentes(leer_etiquetas(rutas.etiquetas)):
        return
    guardar_etiqueta(rutas.etiquetas, recorte, foto, clase)
    st.session_state["mensaje"] = f"Guardado: {recorte} → {nombre_clase}"


def _al_deshacer(rutas) -> None:
    e = deshacer_ultima(rutas.etiquetas)
    st.session_state["mensaje"] = (
        f"Deshecho: {e.recorte} ({e.clase}) vuelve a la cola" if e else "No hay nada que deshacer")


def _guardar_correccion(rutas, recorte: str, foto: str, clase: str, nombre_clase: str) -> None:
    """Cambia la clase de un recorte ya etiquetado: se agrega una fila nueva y vale la última."""
    guardar_etiqueta(rutas.etiquetas, recorte, foto, clase)
    st.session_state["mensaje"] = f"Corregido: {recorte} → {nombre_clase}"


@st.dialog("Corregir etiqueta", width="large")
def _dialogo_corregir(rutas, reg, clase_actual: str, grupos, nombres) -> None:
    """Ventana emergente: un clic en la clase correcta guarda el cambio y la cierra."""
    izq, der = st.columns([1, 2])
    with izq:
        try:
            with Image.open(rutas.recortes / reg.recorte) as im:
                st.image(_ampliar(im), caption=f"Ahora: {nombres.get(clase_actual, clase_actual)}")
        except (OSError, ValueError):
            st.error("No se pudo abrir el recorte.")
        with st.expander("Ver foto completa con la caja"):
            foto = _foto_con_caja(rutas.fotos / reg.foto, reg.caja)
            if foto is None:
                st.warning(f"No se encontró la foto {reg.foto}.")
            else:
                st.image(foto, caption=reg.foto)
    with der:
        st.subheader("¿Qué es?")
        pestanas = st.tabs([TITULOS_GRUPO[g] for g in grupos])
        for pestana, (grupo, clases) in zip(pestanas, grupos.items()):
            with pestana:
                cols = st.columns(COLUMNAS_BOTONES)
                for i, c in enumerate(clases):
                    if cols[i % COLUMNAS_BOTONES].button(
                            c.nombre, key=f"rev_clase_{c.id}", width="stretch",
                            type="primary" if c.id == clase_actual else "secondary"):
                        _guardar_correccion(rutas, reg.recorte, reg.foto, c.id, c.nombre)
                        st.rerun()


def _revisar(rutas, indice, etiquetas, grupos) -> None:
    """Galería de recortes ya etiquetados: pulsa 'Corregir' bajo cualquiera para cambiarlo."""
    vigentes = etiquetas_vigentes(etiquetas)
    if not vigentes:
        return
    nombres = {c.id: c.nombre for g in grupos.values() for c in g}
    nombres.setdefault(CLASE_OMITIDA, "Omitida")
    por_nombre = {r.recorte: r for r in indice}
    with st.expander(f"Revisar y corregir etiquetas anteriores ({len(vigentes)})"):
        presentes = sorted({e.clase for e in vigentes.values()}, key=lambda c: nombres.get(c, c))
        filtro = st.selectbox(
            "Ver solo la clase", ["(todas)"] + presentes,
            format_func=lambda c: c if c == "(todas)" else f"{nombres.get(c, c)}", key="rev_filtro")
        # Las más recientes primero.
        lista = sorted((e for e in vigentes.values() if filtro in ("(todas)", e.clase)),
                       key=lambda e: e.marca_tiempo, reverse=True)
        if not lista:
            st.caption("No hay recortes en esa clase.")
            return
        c_tam, c_pag = st.columns(2)
        por_pagina = c_tam.selectbox("Recortes por página", [30, 60, 100, 200], key="rev_tam")
        paginas = max(1, -(-len(lista) // por_pagina))
        pagina = c_pag.number_input(
            f"Página (de {paginas})", min_value=1, max_value=paginas, value=1, step=1,
            key="rev_pagina")
        ini = (int(pagina) - 1) * por_pagina
        st.caption(f"{len(lista)} recorte(s), del más reciente al más antiguo; "
                   f"mostrando {ini + 1}–{min(ini + por_pagina, len(lista))}. "
                   "Pulsa «Corregir» bajo el que quieras cambiar.")
        galeria = lista[ini:ini + por_pagina]
        cols = st.columns(COLUMNAS_GALERIA)
        for k, e in enumerate(galeria):
            celda = cols[k % COLUMNAS_GALERIA]
            try:
                with Image.open(rutas.recortes / e.recorte) as im:
                    celda.image(im.convert("RGB").copy(), width=100)
            except (OSError, ValueError):
                celda.caption("(sin imagen)")
            celda.caption(nombres.get(e.clase, e.clase)[:24])
            reg = por_nombre.get(e.recorte)
            if celda.button("Corregir", key=f"rev_sel_{e.recorte}", disabled=reg is None):
                _dialogo_corregir(rutas, reg, e.clase, grupos, nombres)


def _instrucciones(rutas) -> None:
    st.info(
        "Todavía no hay recortes para etiquetar.\n\n"
        f"1. Copia tus fotos (jpg, jpeg o png) a `{rutas.fotos}`.\n"
        "2. Corre `python scripts/extraer_recortes.py`.\n"
        "3. Vuelve a cargar esta página."
    )


def _barra_lateral(rutas, indice, etiquetas, grupos) -> None:
    conteos = conteos_por_clase(etiquetas)
    nombres = {c.id: c.nombre for g in grupos.values() for c in g}
    entrenables = ids_de_entrenamiento(_catalogo())
    with st.sidebar:
        st.header("Progreso por clase")
        st.caption(
            f"Fotos distintas que aportan: **{fotos_distintas(etiquetas)}** de "
            f"{len({r.foto for r in indice})}")
        filas = [{"Clase": nombres.get(c, c), "Ejemplos": n} for c, n in conteos.items()]
        if filas:
            st.dataframe(filas, hide_index=True, use_container_width=True)
        else:
            st.caption("Aún no hay etiquetas.")
        pocas = [c for c in clases_con_pocos(conteos, conteos, META_MINIMA)]
        if pocas:
            st.warning(
                f"Menos de {META_MINIMA} ejemplos (meta mínima para entrenar): "
                + ", ".join(f"{nombres.get(c, c)} ({conteos[c]})" for c in pocas))
        sin_ver = [c for c in entrenables if c not in conteos]
        if sin_ver:
            with st.expander(f"Clases sin ningún ejemplo ({len(sin_ver)})"):
                st.write(", ".join(nombres.get(c, c) for c in sin_ver))
        omitidas = sum(1 for e in etiquetas_vigentes(etiquetas).values()
                       if e.clase == CLASE_OMITIDA)
        if omitidas:
            st.caption(f"Omitidas (no cuentan para entrenar): {omitidas}")
        st.divider()
        st.caption(
            "Privacidad: tus fotos y recortes quedan en `datasets/` (ignorado por git, "
            "no se suben).")


def main() -> None:
    st.set_page_config(page_title="Etiquetador de señales", layout="wide")
    st.title("Etiquetador de señales")
    rutas = rutas_por_defecto()
    indice = leer_indice(rutas.indice)
    if not indice:
        _instrucciones(rutas)
        st.caption("Privacidad: las fotos y recortes quedan en `datasets/` (ignorado por git).")
        return

    grupos = clases_disponibles(_catalogo())
    etiquetas = leer_etiquetas(rutas.etiquetas)
    vigentes = etiquetas_vigentes(etiquetas)
    hechos = sum(1 for r in indice if r.recorte in vigentes)
    _barra_lateral(rutas, indice, etiquetas, grupos)

    st.progress(hechos / len(indice), text=f"Etiquetados: {hechos} / {len(indice)}")
    mensaje = st.session_state.pop("mensaje", None)
    if mensaje:
        st.success(mensaje)

    _revisar(rutas, indice, etiquetas, grupos)

    actual = siguiente_sin_etiquetar(indice, etiquetas)
    if actual is None:
        st.balloons()
        st.success("No quedan recortes por etiquetar. Puedes deshacer la última si te equivocaste.")
        st.button("Deshacer última etiqueta", on_click=_al_deshacer, args=(rutas,),
                  disabled=not etiquetas)
        return

    izq, der = st.columns([1, 2])
    with izq:
        ruta_recorte = rutas.recortes / actual.recorte
        try:
            with Image.open(ruta_recorte) as im:
                st.image(_ampliar(im), caption=f"{actual.recorte} · puntaje {actual.puntaje:.2f}")
        except (OSError, ValueError):
            st.error(f"No se pudo abrir el recorte {ruta_recorte}. Omítelo para continuar.")
        with st.expander("Ver foto completa con la caja"):
            foto = _foto_con_caja(rutas.fotos / actual.foto, actual.caja)
            if foto is None:
                st.warning(f"No se encontró la foto {actual.foto} en {rutas.fotos}.")
            else:
                st.image(foto, caption=actual.foto)
        st.button("Deshacer última etiqueta", on_click=_al_deshacer, args=(rutas,),
                  disabled=not etiquetas, key="deshacer")
    with der:
        st.subheader("¿Qué es?")
        pestanas = st.tabs([TITULOS_GRUPO[g] for g in grupos])
        for pestana, (grupo, clases) in zip(pestanas, grupos.items()):
            with pestana:
                if not clases:
                    st.caption("El catálogo aún no tiene señales de este grupo; usa "
                               "'Otra señal' en la pestaña Especiales.")
                cols = st.columns(COLUMNAS_BOTONES)
                for i, c in enumerate(clases):
                    cols[i % COLUMNAS_BOTONES].button(
                        c.nombre, key=f"clase_{c.id}", use_container_width=True,
                        type="primary" if c.id == "no_senal" else "secondary",
                        on_click=_al_etiquetar,
                        args=(rutas, actual.recorte, actual.foto, c.id, c.nombre))


main()
