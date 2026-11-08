import base64
import dataclasses
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageFont, ImageOps

from roadvision.config import Config
from roadvision.contracts import Categoria, Estado
from roadvision.pipeline import ETAPAS, construir_pipeline

st.set_page_config(page_title="RoadVision", page_icon="🚦", layout="wide")

st.markdown(
    """
    <style>
    [data-testid="stSidebar"] { background-color: #4F4F4F; }
    [data-testid="stSidebar"] * { color: #E8F3EC; }
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

NOMBRE_ETAPA = {
    "total": "Total",
    "preproceso": "Preproceso",
    "deteccion": "Detección",
    "clasificacion": "Clasificación",
    "respaldo": "Respaldo CLIP",
}

LOGO_PNG = Path(__file__).parent / "assets" / "logo.png"
LOGO_SVG = (
    "<svg xmlns='http://www.w3.org/2000/svg' width='64' height='64' viewBox='0 0 64 64'>"
    "<polygon points='32,3 61,32 32,61 3,32' fill='#F2B800' stroke='#111' stroke-width='3'/>"
    "<rect x='29' y='16' width='6' height='20' rx='2' fill='#111'/>"
    "<circle cx='32' cy='45' r='4' fill='#111'/></svg>"
)


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


@st.cache_resource(show_spinner="Cargando modelos en CPU para comparar...")
def cargar_pipeline_cpu():
    config = dataclasses.replace(Config(), dispositivo="cpu", fp16=False)
    pipeline = construir_pipeline(config)
    pipeline.calentar()
    return pipeline


def medir(pipe, imagen, umbrales, repeticiones=3):
    acumulado = {e: 0.0 for e in ETAPAS}
    for _ in range(repeticiones):
        r = pipe.analizar(imagen, umbrales)
        for e in ETAPAS:
            acumulado[e] += r.tiempos_ms.get(e, 0.0)
    return {e: v / repeticiones for e, v in acumulado.items()}


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


def leyenda_colores():
    items = [
        ("Reglamentaria", a_hex(COLORES[Categoria.REGLAMENTARIA])),
        ("Preventiva", a_hex(COLORES[Categoria.PREVENTIVA])),
        ("Informativa", a_hex(COLORES[Categoria.INFORMATIVA])),
        ("No reconocida", a_hex(GRIS)),
    ]
    html = "".join(
        f"<span style='margin-right:18px;font-size:0.9rem'>"
        f"<span style='display:inline-block;width:12px;height:12px;"
        f"background:{color};border-radius:2px;margin-right:6px'></span>{nombre}</span>"
        for nombre, color in items
    )
    return f"<div style='margin:6px 0 14px'>{html}</div>"


# ---------- Cargar modelos ----------
pipeline, calentamiento = cargar_pipeline()
base = pipeline.config.umbrales
es_gpu = pipeline.dispositivo.nombre == "cuda"

# ---------- Barra lateral ----------
with st.sidebar:
    if LOGO_PNG.exists():
        st.image(str(LOGO_PNG), use_container_width=True)
    else:
        logo64 = base64.b64encode(LOGO_SVG.encode()).decode()
        st.markdown(
            f"<div style='display:flex;align-items:center;gap:10px;margin-bottom:8px'>"
            f"<img src='data:image/svg+xml;base64,{logo64}' width='46'>"
            f"<span style='font-size:1.4rem;font-weight:700;color:#F2B800'>RoadVision</span>"
            f"</div>",
            unsafe_allow_html=True,
        )
    st.divider()
    st.header("Ajustes")
    umbral_det = st.slider("Umbral de detección", 0.05, 0.90, base.deteccion, 0.05)
    umbral_cls = st.slider("Umbral de clasificación", 0.50, 0.99, base.gtsrb, 0.01)
    mostrar_no = st.checkbox("Mostrar no reconocidas", value=True)
    st.divider()
    st.caption("Dispositivo en uso")
    st.markdown(
        etiqueta("GPU" if es_gpu else "CPU", "#76B900" if es_gpu else "#6B8F7B"),
        unsafe_allow_html=True,
    )
    st.write(pipeline.dispositivo.descripcion)

umbrales = dataclasses.replace(base, deteccion=umbral_det, gtsrb=umbral_cls)

# ---------- Banner ----------
st.markdown(
    "<div style='background:linear-gradient(90deg,#4F4F4F,#707070);"
    "padding:18px 24px;border-radius:12px;margin-bottom:14px'>"
    "<span style='font-size:2rem;font-weight:700;color:#D43131'>🚦 RoadVision</span><br>"
    "<span style='color:#E8F3EC'>Sube una foto de una vía y detecto sus señales de tránsito.</span>"
    "</div>",
    unsafe_allow_html=True,
)

archivo = st.file_uploader("Foto de la vía", type=["jpg", "jpeg", "png"])

if archivo:
    imagen = ImageOps.exif_transpose(Image.open(archivo)).convert("RGB")

    # Se analiza solo cuando cambia la foto o los umbrales (no en cada clic)
    clave = (archivo.name, archivo.size, umbral_det, umbral_cls)
    if st.session_state.get("clave") != clave:
        with st.spinner("Analizando..."):
            st.session_state["resultado"] = pipeline.analizar(imagen, umbrales)
        st.session_state["clave"] = clave
    resultado = st.session_state["resultado"]

    senales = [
        s for s in resultado.senales
        if mostrar_no or s.estado is not Estado.NO_RECONOCIDA
    ]

    tab_res, tab_sen, tab_rend, tab_comp = st.tabs(
        ["Resultado", "Señales", "Rendimiento", "CPU vs GPU"]
    )

    # ----- Pestaña 1: Resultado -----
    with tab_res:
        if not senales:
            st.info("No encontré señales de tránsito en esta foto.")
        else:
            st.image(dibujar_cajas(imagen, senales), use_container_width=True)
            st.markdown(leyenda_colores(), unsafe_allow_html=True)

            st.subheader("Resumen")
            por_categoria = {}
            for s in senales:
                clave_cat = s.categoria.value if s.categoria else "sin categoría"
                por_categoria[clave_cat] = por_categoria.get(clave_cat, 0) + 1

            columnas = st.columns(len(por_categoria) + 1)
            columnas[0].metric("Total de señales", len(senales))
            for col, (nombre, cantidad) in zip(columnas[1:], por_categoria.items()):
                col.metric(nombre.capitalize(), cantidad)

            distintas = {s.nombre: s.significado for s in senales if s.significado}
            for nombre, significado in distintas.items():
                st.markdown(f"- **{nombre}**: {significado}")

    # ----- Pestaña 2: Señales -----
    with tab_sen:
        if not senales:
            st.info("No hay señales para mostrar.")
        for s in senales:
            c1, c2 = st.columns([1, 5])
            c1.image(s.recorte, width=90)
            categoria = s.categoria.value if s.categoria else "sin categoría"
            c2.markdown(
                f"**{s.numero}. {s.nombre}**<br>"
                + etiqueta(categoria, a_hex(color_de(s)))
                + etiqueta(ETIQUETA_ESTADO[s.estado], COLOR_ESTADO[s.estado])
                + f"<span style='opacity:0.8'>confianza {s.confianza:.0%}</span>",
                unsafe_allow_html=True,
            )
            if s.significado:
                c2.write(s.significado)

    # ----- Pestaña 3: Rendimiento -----
    with tab_rend:
        total = resultado.tiempos_ms.get("total", 0.0)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Dispositivo", "GPU" if es_gpu else "CPU")
        c1.caption(resultado.dispositivo)
        c2.metric("Tiempo total", f"{total:.0f} ms")
        c3.metric("Imágenes por segundo", f"{1000 / total:.2f}" if total > 0 else "—")
        c4.metric("Señales detectadas", len(resultado.senales))

        st.subheader("Tiempo por etapa")
        etapas = [e for e in ETAPAS if e != "total"]
        st.bar_chart(
            pd.DataFrame(
                {"Tiempo (ms)": [resultado.tiempos_ms.get(e, 0.0) for e in etapas]},
                index=[NOMBRE_ETAPA[e] for e in etapas],
            ),
            color="#F2B800",
        )
        st.caption("El calentamiento inicial no entra en estos tiempos.")

        st.subheader("Carga de modelos")
        nombres = {"detector": "Detector (OWLv2)", "clasificador": "Clasificador (ViT)",
                   "respaldo": "Respaldo (CLIP)"}
        st.dataframe(
            {
                "Modelo": [nombres.get(k, k) for k in pipeline.carga],
                "Carga (ms)": [round(c.tiempo_ms) for c in pipeline.carga.values()],
                "VRAM (MB)": [round(c.vram_mb) for c in pipeline.carga.values()],
            },
            hide_index=True,
        )
        if not es_gpu:
            st.caption("En CPU la VRAM es 0: no se usa memoria de la GPU.")

    # ----- Pestaña 4: CPU vs GPU -----
    with tab_comp:
        if not es_gpu:
            st.info(
                "Este equipo no está usando una GPU NVIDIA con CUDA, así que no hay "
                "nada que comparar. Ejecuta la app en el PC con GPU para activar esta pestaña."
            )
        else:
            st.write(
                "Analiza la foto actual en GPU y en CPU (3 repeticiones cada una, "
                "sin contar el calentamiento) y compara los tiempos."
            )
            if st.button("Comparar CPU y GPU"):
                with st.spinner("Midiendo en GPU y en CPU..."):
                    cpu = cargar_pipeline_cpu()
                    t_gpu = medir(pipeline, imagen, umbrales)
                    t_cpu = medir(cpu, imagen, umbrales)
                st.session_state["comparacion"] = {"clave": clave, "cpu": t_cpu, "gpu": t_gpu}

            comp = st.session_state.get("comparacion")
            if comp and comp["clave"] == clave:
                t_cpu, t_gpu = comp["cpu"], comp["gpu"]
                aceleracion = t_cpu["total"] / t_gpu["total"] if t_gpu["total"] > 0 else 0
                c1, c2, c3 = st.columns(3)
                c1.metric("CPU (total)", f"{t_cpu['total']:.0f} ms")
                c2.metric("GPU (total)", f"{t_gpu['total']:.0f} ms")
                c3.metric("Aceleración", f"{aceleracion:.1f}×")

                etapas = [e for e in ETAPAS if e != "total"]
                tabla = pd.DataFrame(
                    {"CPU": [t_cpu[e] for e in etapas], "GPU": [t_gpu[e] for e in etapas]},
                    index=[NOMBRE_ETAPA[e] for e in etapas],
                )
                st.bar_chart(tabla, color=["#6B8F7B", "#76B900"])
                st.dataframe(tabla.round(1))
else:
    st.info("Sube una foto para empezar.")