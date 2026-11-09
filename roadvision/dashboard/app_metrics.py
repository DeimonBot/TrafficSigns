"""Monitor de uso del computador en vivo (spec §5.5/§5.6), independiente de los modelos.

Ejecutar: `streamlit run roadvision/dashboard/app_metrics.py`

Usa `MuestreadorSistema` de `roadvision.metrics` (CPU, RAM, GPU, VRAM, temperatura,
potencia). Muestra los últimos ~60 s con refresco periódico, un botón de pausa, el
indicador GPU/CPU y, debajo, el histórico del CSV con `render_panel`.

Alcance de las magnitudes: CPU % y RAM en vivo son del PROCESO DEL MONITOR (Streamlit),
no del equipo ni de la app que analiza; GPU/VRAM/temperatura/potencia son del
dispositivo completo. El CPU/RAM de la app que analiza está en el histórico.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Permite `streamlit run` desde cualquier carpeta sin instalar el paquete.
_RAIZ = Path(__file__).resolve().parents[2]
if str(_RAIZ) not in sys.path:
    sys.path.insert(0, str(_RAIZ))

import streamlit as st  # noqa: E402

from roadvision.dashboard.panel import mostrar_figura  # noqa: E402

VENTANA_S = 60.0
INTERVALO_MUESTREO_S = 0.5
# El doble de la ventana visible: acota la memoria del muestreador, que vive mientras
# viva el servidor (sin límite crecería ~170 mil muestras al día).
MAX_MUESTRAS = int(2 * VENTANA_S / INTERVALO_MUESTREO_S)
REFRESCO_S = 1.0

# (atributo, título, color, unidad); un gráfico por magnitud, sin doble eje.
SERIES_VIVO = (
    ("cpu_pct", "CPU % (proceso del monitor)", "#CC79A7", "%"),
    ("ram_proceso_mb", "RAM (proceso del monitor)", "#56B4E9", "MB"),
    ("gpu_pct", "GPU % (dispositivo)", "#0072B2", "%"),
    ("vram_usada_mb", "VRAM usada (dispositivo)", "#E69F00", "MB"),
    ("temperatura_c", "Temperatura", "#009E73", "°C"),
    ("potencia_w", "Potencia", "#D55E00", "W"),
)


@st.cache_resource(show_spinner=False)
def _crear_muestreador():
    """Un único muestreador por proceso del servidor; se mantiene abierto."""
    from roadvision.metrics import MuestreadorSistema

    muestreador = MuestreadorSistema(INTERVALO_MUESTREO_S, max_muestras=MAX_MUESTRAS)
    muestreador.__enter__()
    return muestreador


def _figura_serie(muestras, t_ref, attr, titulo, color, unidad):
    import plotly.graph_objects as go

    from roadvision.dashboard.panel import _estilo

    x = [m.t_s - t_ref for m in muestras]
    y = [getattr(m, attr) for m in muestras]
    fig = go.Figure(go.Scatter(
        x=x, y=y, mode="lines", line=dict(color=color, width=2), showlegend=False,
        hovertemplate=f"{titulo}: %{{y:.1f}} {unidad}<br>hace %{{x:.0f}} s<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text=f"{titulo} ({unidad})", font=dict(size=13)),
        xaxis=dict(range=[-VENTANA_S, 0], title="segundos (0 = ahora)"),
        hovermode="closest",
    )
    return _estilo(fig, altura=220)


def _vista_en_vivo(pausado: bool) -> None:
    """Dibuja tarjetas y gráficas; con la pausa activa muestra la foto congelada."""
    try:
        muestreador = _crear_muestreador()
        todas = list(muestreador.muestras)
        gpu = bool(muestreador.gpu_disponible)
    except NotImplementedError:
        st.error("El muestreador de `roadvision.metrics` aún no está implementado.")
        return
    except Exception as exc:
        st.error(f"No se pudo iniciar el muestreo del sistema: {exc}")
        return
    if not todas:
        st.info("Esperando las primeras muestras...")
        return

    if pausado and "congelado" in st.session_state:
        ventana = st.session_state["congelado"]
    else:
        t_fin = todas[-1].t_s
        ventana = [m for m in todas if m.t_s >= t_fin - VENTANA_S]
        if pausado:
            st.session_state["congelado"] = ventana
    if not ventana:
        return
    t_corte = ventana[-1].t_s
    ultima = ventana[-1]

    estado = "GPU NVIDIA detectada" if gpu else "CPU (sin GPU NVIDIA)"
    st.markdown(f"**Dispositivo de cómputo:** {estado}" + ("  ·  *en pausa*" if pausado else ""))
    st.caption("CPU y RAM en vivo son del proceso del monitor; GPU/VRAM/temperatura/potencia "
               "son del dispositivo completo. El CPU/RAM de la app que analiza está en el "
               "histórico.")
    cols = st.columns(4 if gpu else 2)
    cols[0].metric("CPU % (proceso del monitor)", f"{ultima.cpu_pct:.0f} %",
                   help="Solo este proceso de Streamlit; puede superar 100 %. "
                        "NO incluye la app que analiza.")
    cols[1].metric("RAM (proceso del monitor)", f"{ultima.ram_proceso_mb:,.0f} MB",
                   help="Solo este proceso de Streamlit. NO incluye la app que analiza.")
    if gpu:
        gpu_pct = ultima.gpu_pct
        cols[2].metric("GPU % (dispositivo)", "—" if gpu_pct is None else f"{gpu_pct:.0f} %",
                       help="Uso del dispositivo completo, incluye otros procesos.")
        vram = ultima.vram_usada_mb
        cols[3].metric("VRAM usada (dispositivo)", "—" if vram is None else f"{vram:,.0f} MB")

    series = [s for s in SERIES_VIVO
              if any(getattr(m, s[0]) is not None for m in ventana)]
    for i in range(0, len(series), 2):
        fila = st.columns(2)
        for col, (attr, titulo, color, unidad) in zip(fila, series[i:i + 2]):
            with col:
                mostrar_figura(
                    st, _figura_serie(ventana, t_corte, attr, titulo, color, unidad),
                    f"vivo_{attr}",
                )


def main() -> None:
    st.set_page_config(page_title="RoadVision · Rendimiento", layout="wide")
    st.title("Rendimiento del equipo")
    st.caption("Monitor en vivo del sistema (no necesita cargar los modelos) e histórico "
               "de los análisis.")

    pausado = st.toggle("Pausar monitor", value=False, key="pausa")
    if not pausado:
        st.session_state.pop("congelado", None)
    # Con la pausa se quita el refresco automático; sin ella se redibuja cada segundo.
    st.fragment(run_every=None if pausado else REFRESCO_S)(_vista_en_vivo)(pausado)

    st.divider()
    st.subheader("Histórico de análisis")
    from roadvision.dashboard.panel import render_panel

    render_panel()


main()
