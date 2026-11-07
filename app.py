import dataclasses

import streamlit as st
from PIL import Image, ImageDraw, ImageFont, ImageOps

from roadvision.config import Config
from roadvision.contracts import Categoria, Estado
from roadvision.pipeline import construir_pipeline

st.set_page_config(page_title="RoadVision", page_icon="🚦", layout="wide")

COLORES = {
    Categoria.REGLAMENTARIA: (220, 40, 40),   # rojo
    Categoria.PREVENTIVA: (240, 190, 0),      # amarillo
    Categoria.INFORMATIVA: (40, 90, 220),     # azul
}
GRIS = (130, 130, 130)

ETIQUETA_ESTADO = {
    Estado.RECONOCIDA: "Reconocida",
    Estado.APROXIMADA: "Aproximada",
    Estado.SOLO_CATEGORIA: "Solo categoría",
    Estado.NO_RECONOCIDA: "No reconocida",
}


@st.cache_resource(show_spinner="Cargando modelos (solo la primera vez)...")
def cargar_pipeline():
    pipeline = construir_pipeline()
    calentamiento = pipeline.calentar()
    return pipeline, calentamiento


def color_de(senal):
    if senal.estado is Estado.NO_RECONOCIDA or senal.categoria is None:
        return GRIS
    return COLORES.get(senal.categoria, GRIS)


def dibujar_cajas(imagen, senales):
    img = imagen.copy()
    dib = ImageDraw.Draw(img)
    grosor = max(3, img.width // 300)
    fuente = ImageFont.load_default(size=max(16, img.width // 50))
    for s in senales:
        color = color_de(s)
        dib.rectangle(s.caja, outline=color, width=grosor)
        x, y = s.caja[0], s.caja[1]
        dib.rectangle((x, y, x + grosor * 8, y + grosor * 8), fill=color)
        dib.text((x + 3, y), str(s.numero), fill=(255, 255, 255), font=fuente)
    return img


# ---------- Cargar modelos ----------
pipeline, calentamiento = cargar_pipeline()
base = pipeline.config.umbrales

# ---------- Barra lateral ----------
with st.sidebar:
    st.header("Ajustes")
    umbral_det = st.slider("Umbral de detección", 0.05, 0.90, base.deteccion, 0.05)
    umbral_cls = st.slider("Umbral de clasificación", 0.50, 0.99, base.gtsrb, 0.01)
    mostrar_no = st.checkbox("Mostrar no reconocidas", value=True)
    mostrar_rend = st.checkbox("Mostrar panel de rendimiento", value=False)
    st.divider()
    st.caption("Dispositivo en uso")
    st.write(pipeline.dispositivo.descripcion)

umbrales = dataclasses.replace(base, deteccion=umbral_det, gtsrb=umbral_cls)

# ---------- Pantalla principal ----------
st.title("🚦 RoadVision")
st.caption("Sube una foto de una vía y detecto sus señales de tránsito.")

archivo = st.file_uploader("Foto de la vía", type=["jpg", "jpeg", "png"])

if archivo:
    imagen = ImageOps.exif_transpose(Image.open(archivo)).convert("RGB")
    with st.spinner("Analizando..."):
        resultado = pipeline.analizar(imagen, umbrales)

    senales = [
        s for s in resultado.senales
        if mostrar_no or s.estado is not Estado.NO_RECONOCIDA
    ]

    if not senales:
        st.info("No encontré señales de tránsito en esta foto.")
    else:
        st.image(dibujar_cajas(imagen, senales), use_container_width=True)

        st.subheader("Señales encontradas")
        for s in senales:
            c1, c2 = st.columns([1, 5])
            c1.image(s.recorte, width=90)
            categoria = s.categoria.value if s.categoria else "sin categoría"
            c2.markdown(
                f"**{s.numero}. {s.nombre}** · {categoria} · "
                f"{ETIQUETA_ESTADO[s.estado]} · confianza {s.confianza:.0%}"
            )
            if s.significado:
                c2.write(s.significado)

        st.subheader("Resumen")
        st.write(f"**Total de señales:** {len(senales)}")
        por_categoria = {}
        for s in senales:
            clave = s.categoria.value if s.categoria else "sin categoría"
            por_categoria[clave] = por_categoria.get(clave, 0) + 1
        st.write("**Por categoría:** " + ", ".join(
            f"{k}: {v}" for k, v in por_categoria.items()))

        distintas = {}
        for s in senales:
            if s.significado:
                distintas[s.nombre] = s.significado
        for nombre, significado in distintas.items():
            st.markdown(f"- **{nombre}**: {significado}")

    if mostrar_rend:
        with st.expander("Rendimiento", expanded=True):
            st.write(f"Dispositivo: {resultado.dispositivo}")
            st.dataframe(
                {"Etapa": list(resultado.tiempos_ms.keys()),
                 "Tiempo (ms)": [round(v, 1) for v in resultado.tiempos_ms.values()]},
                hide_index=True,
            )
            st.caption("El calentamiento inicial no entra en estos tiempos.")