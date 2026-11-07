import dataclasses

import streamlit as st
from PIL import Image, ImageDraw, ImageFont, ImageOps

from roadvision.config import Config
from roadvision.contracts import Categoria, Estado
from roadvision.pipeline import construir_pipeline

st.set_page_config(page_title="RoadVision", page_icon="🚦", layout="wide")

st.markdown(
    """
    <style>
    [data-testid="stSidebar"] { background-color: #444444; }
    [data-testid="stSidebar"] * { color: #EAF2FA; }
    h1 { color: #F2B800; }
    </style>
    """,
    unsafe_allow_html=True,
)

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

COLOR_ESTADO = {
    Estado.RECONOCIDA: "#2E9E5B",      # verde
    Estado.APROXIMADA: "#E67E22",      # naranja
    Estado.SOLO_CATEGORIA: "#8E5BD9",  # morado
    Estado.NO_RECONOCIDA: "#828282",   # gris
}


def a_hex(rgb):
    return "#%02x%02x%02x" % rgb


def etiqueta(texto, color_hex):
    r, g, b = (int(color_hex[i:i + 2], 16) for i in (1, 3, 5))
    texto_color = "#222222" if (r * 299 + g * 587 + b * 114) / 1000 > 160 else "#FFFFFF"
    return (
        f"<span style='background:{color_hex};color:{texto_color};"
        "padding:2px 10px;border-radius:12px;font-size:0.8rem;"
        f"font-weight:600;margin-right:6px'>{texto}</span>"
    )


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
        lado = grosor * 8
        y_et = max(0, y - lado)          # la etiqueta va encima de la caja
        dib.rectangle((x, y_et, x + lado, y_et + lado), fill=color)
        dib.text((x + 3, y_et), str(s.numero), fill=(255, 255, 255), font=fuente)
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
            color_cat = color_de(s)
            c2.markdown(
                f"**{s.numero}. {s.nombre}**<br>"
                + etiqueta(categoria, a_hex(color_cat))
                + etiqueta(ETIQUETA_ESTADO[s.estado], COLOR_ESTADO[s.estado])
                + f"<span style='opacity:0.8'>confianza {s.confianza:.0%}</span>",
                unsafe_allow_html=True,
            )
            if s.significado:
                c2.write(s.significado)

        st.subheader("Resumen")
        por_categoria = {}
        for s in senales:
            clave = s.categoria.value if s.categoria else "sin categoría"
            por_categoria[clave] = por_categoria.get(clave, 0) + 1

        columnas = st.columns(len(por_categoria) + 1)
        columnas[0].metric("Total de señales", len(senales))
        for col, (nombre, cantidad) in zip(columnas[1:], por_categoria.items()):
            col.metric(nombre.capitalize(), cantidad)

        distintas = {s.nombre: s.significado for s in senales if s.significado}
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