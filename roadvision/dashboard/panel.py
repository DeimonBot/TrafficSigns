"""Panel de rendimiento reutilizable (spec §5.6), pensado para incrustarse en `app.py`.

Uso: `render_panel()` (lee `logs/metricas.csv`) o `render_panel(ruta)`.

Etiquetas honestas (ver contrato de `roadvision.metrics`):
- "imágenes/s equivalente" = 1000 / mediana de ms_total, no throughput real.
- GPU % y VRAM (pico) son del DISPOSITIVO completo (incluye otros procesos).
- Las tarjetas salen todas de la MISMA fila (el último análisis).
- CPU % puede superar 100 % (suma varios núcleos).
Sin GPU NVIDIA los campos vienen vacíos y se muestra "CPU" sin errores.

Colores: paleta Okabe-Ito (segura para daltonismo, sin rojo/verde como par), con fondo
transparente y texto del tema de Streamlit para funcionar en modo claro y oscuro.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

# Etapas en el orden en que ocurren; el color sigue a la etapa, nunca a su rango.
ETAPAS_PANEL = (
    ("ms_preproceso", "Preproceso", "#56B4E9"),
    ("ms_deteccion", "Detección", "#0072B2"),
    ("ms_clasificacion", "Clasificación", "#E69F00"),
    ("ms_respaldo", "Respaldo CLIP", "#CC79A7"),
)
COLOR_GPU = "#0072B2"
COLOR_VRAM = "#E69F00"
COLOR_CPU = "#CC79A7"
COLOR_TOTAL = "#0072B2"
COLOR_REF = "#999999"
_REJILLA = "rgba(128,128,128,0.25)"
_MAX_ANALISIS = 40  # barras visibles en las gráficas por análisis
MIN_P95 = 20  # mínimo de análisis para mostrar p95

_COLUMNAS_NUMERICAS = (
    "ms_total", "ms_preproceso", "ms_deteccion", "ms_clasificacion", "ms_respaldo",
    "imagenes_por_s", "gpu_pct_pico", "gpu_pct_prom", "vram_usada_mb",
    "vram_pico_torch_mb", "temperatura_c_max", "potencia_w_prom", "cpu_pct_prom",
    "ram_proceso_mb_pico",
)


def _ruta_por_defecto() -> Path:
    from roadvision.metrics import RUTA_CSV

    ruta = Path(RUTA_CSV)
    return ruta if ruta.is_absolute() else Path(__file__).resolve().parents[2] / ruta


def leer_datos(ruta: Path) -> pd.DataFrame:
    """Lee el CSV con `metrics.leer_csv`; si aún no está implementado, usa pandas.

    Siempre excluye el calentamiento y devuelve columnas numéricas ya convertidas.
    """
    import pandas as pd

    try:
        from roadvision.metrics import leer_csv

        df = leer_csv(ruta)
    except (NotImplementedError, ImportError):
        df = _leer_csv_respaldo(ruta)
    for col in _COLUMNAS_NUMERICAS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.reset_index(drop=True)


def _leer_csv_respaldo(ruta: Path) -> pd.DataFrame:
    import pandas as pd

    try:
        from roadvision.metrics import COLUMNAS_CSV
    except ImportError:
        COLUMNAS_CSV = None
    if not ruta.exists():
        return pd.DataFrame(columns=COLUMNAS_CSV)
    df = pd.read_csv(ruta)
    if "calentamiento" in df.columns:
        calent = df["calentamiento"].astype(str).str.lower().isin(("true", "1"))
        df = df[~calent]
    return df


def _estilo(fig, altura: int = 300):
    """Aplica estilo común: fondo transparente, rejilla discreta, leyenda arriba."""
    fig.update_layout(
        height=altura,
        margin=dict(l=8, r=8, t=36, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        hovermode="x unified",
    )
    fig.update_xaxes(gridcolor=_REJILLA, zeroline=False)
    fig.update_yaxes(gridcolor=_REJILLA, zeroline=False, rangemode="tozero")
    return fig


def mostrar_figura(st, fig, clave: str) -> None:
    """`st.plotly_chart` a todo el ancho (Streamlit 1.67 fijado en requirements.txt)."""
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False}, key=clave)


def _valor(df: pd.DataFrame, col: str) -> float | None:
    """Valor de la ÚLTIMA fila (misma fila para todas las tarjetas); None si vacío."""
    if col not in df.columns or df.empty:
        return None
    v = df[col].iloc[-1]
    return None if v != v else float(v)  # NaN != NaN


def _fmt(valor: float | None, unidad: str, decimales: int = 0) -> str:
    return "—" if valor is None else f"{valor:,.{decimales}f} {unidad}".strip()


def hay_gpu(df: pd.DataFrame) -> bool:
    """True si alguna fila trae métricas de GPU (campos no vacíos)."""
    return "gpu_pct_pico" in df.columns and bool(df["gpu_pct_pico"].notna().any())


def _tarjetas(st, df: pd.DataFrame, con_gpu: bool) -> None:
    dispositivo = str(df["dispositivo"].fillna("").iloc[-1]) if "dispositivo" in df.columns else ""
    precision = str(df["precision"].fillna("").iloc[-1]) if "precision" in df.columns else ""
    marca = df["marca_tiempo"].iloc[-1] if "marca_tiempo" in df.columns else "?"
    st.caption(
        f"Último análisis ({marca}) · dispositivo: "
        f"{dispositivo or 'desconocido'} {precision}".rstrip()
    )
    cols = st.columns(6)
    if con_gpu:
        cols[0].metric(
            "GPU % pico (dispositivo)", _fmt(_valor(df, "gpu_pct_pico"), "%"),
            help="Uso del dispositivo completo durante el análisis; incluye otros procesos.",
        )
        cols[1].metric(
            "VRAM pico (dispositivo)", _fmt(_valor(df, "vram_usada_mb"), "MB"),
            help="Pico de VRAM del dispositivo completo durante el análisis, no solo de esta app.",
        )
        cols[2].metric("Temperatura máx.", _fmt(_valor(df, "temperatura_c_max"), "°C"))
        cols[3].metric("Potencia prom.", _fmt(_valor(df, "potencia_w_prom"), "W", 1))
    else:
        cols[0].metric("GPU", "CPU", help="Sin GPU NVIDIA: no hay métricas de GPU.")
        cols[1].metric("VRAM", "—")
        cols[2].metric("Temperatura", "—")
        cols[3].metric("Potencia", "—")
    cols[4].metric(
        "CPU % prom. (proceso que analiza)", _fmt(_valor(df, "cpu_pct_prom"), "%"),
        help="Puede superar 100 % porque suma varios núcleos.",
    )
    cols[5].metric(
        "RAM proceso que analiza (pico)", _fmt(_valor(df, "ram_proceso_mb_pico"), "MB")
    )
    if "ms_total" in df.columns and df["ms_total"].notna().any():
        p50 = float(df["ms_total"].quantile(0.50))
        if p50 > 0:
            st.metric(
                "Imágenes/s equivalente (mediana)", f"{1000 / p50:,.2f}",
                help="1000 / p50 de ms_total; equivalente, no throughput real.",
            )


def _grafica_etapas(go, df: pd.DataFrame):
    base = df.tail(_MAX_ANALISIS)
    x = base.index + 1
    fig = go.Figure()
    for col, nombre, color in ETAPAS_PANEL:
        if col in base.columns and base[col].notna().any():
            fig.add_bar(
                x=x, y=base[col], name=nombre, marker_color=color,
                hovertemplate=f"{nombre}: %{{y:.1f}} ms<extra></extra>",
            )
    fig.update_layout(
        barmode="stack", bargap=0.25,
        title=dict(text="Tiempo por etapa (ms)", font=dict(size=14)),
        xaxis_title="Nº de análisis", yaxis_title="ms",
    )
    return _estilo(fig)


def _grafica_total(go, df: pd.DataFrame, p50: float | None, p95: float | None):
    fig = go.Figure()
    fig.add_scatter(
        x=df.index + 1, y=df["ms_total"], mode="lines+markers", name="ms total",
        line=dict(color=COLOR_TOTAL, width=2), marker=dict(size=8),
        hovertemplate="Total: %{y:.1f} ms<extra></extra>",
    )
    if p50 is not None:
        fig.add_hline(
            y=p50, line=dict(color=COLOR_REF, width=1, dash="dash"),
            annotation_text=f"p50 {p50:.0f} ms", annotation_position="top left",
        )
    if p95 is not None:
        fig.add_hline(
            y=p95, line=dict(color=COLOR_REF, width=1, dash="dot"),
            annotation_text=f"p95 {p95:.0f} ms", annotation_position="top left",
        )
    fig.update_layout(
        title=dict(text="Histórico de tiempo total (ms)", font=dict(size=14)),
        xaxis_title="Nº de análisis", yaxis_title="ms", showlegend=False,
    )
    return _estilo(fig)


def _grafica_recursos(go, df: pd.DataFrame, con_gpu: bool):
    """Una fila por magnitud (cada una con su propio eje): nunca doble eje Y."""
    from plotly.subplots import make_subplots

    filas = []
    if con_gpu:
        filas += [
            ("gpu_pct_pico", "GPU % pico (dispositivo)", COLOR_GPU, "%"),
            ("vram_usada_mb", "VRAM pico (dispositivo)", COLOR_VRAM, "MB"),
        ]
    filas.append(("cpu_pct_prom", "CPU % prom. (proceso que analiza)", COLOR_CPU, "%"))
    df = df.tail(_MAX_ANALISIS)
    fig = make_subplots(
        rows=len(filas), cols=1, shared_xaxes=True, vertical_spacing=0.1,
        subplot_titles=[f[1] for f in filas],
    )
    for i, (col, nombre, color, unidad) in enumerate(filas, start=1):
        fig.add_scatter(
            x=df.index + 1, y=df[col], mode="lines+markers", name=nombre,
            line=dict(color=color, width=2), marker=dict(size=7), showlegend=False,
            hovertemplate=f"{nombre}: %{{y:.1f}} {unidad}<extra></extra>",
            row=i, col=1,
        )
    fig.update_xaxes(title_text="Nº de análisis", row=len(filas), col=1)
    fig.update_annotations(font_size=12, xanchor="left", x=0)
    return _estilo(fig, altura=170 * len(filas) + 60)


def _tabla_percentiles(df: pd.DataFrame):
    import pandas as pd

    filas = []
    for col, nombre in (("ms_total", "Total"),) + tuple((c, n) for c, n, _ in ETAPAS_PANEL):
        if col in df.columns and df[col].notna().any():
            serie = df[col].dropna()
            n = len(serie)
            filas.append({
                "Métrica": f"{nombre} (ms)",
                "n": n,
                "p50": round(float(serie.quantile(0.50)), 1),
                "p95": round(float(serie.quantile(0.95)), 1) if n >= MIN_P95 else None,
            })
    return pd.DataFrame(filas)


def render_panel(csv_path: str | Path | None = None, *, clave: str = "rend") -> None:
    """Dibuja el panel de rendimiento con el histórico del CSV de métricas.

    `csv_path=None` usa `roadvision.metrics.RUTA_CSV`. `clave` evita colisiones de ids de
    widgets si el panel se muestra más de una vez en la misma página.
    """
    import plotly.graph_objects as go
    import streamlit as st

    ruta = Path(csv_path) if csv_path is not None else _ruta_por_defecto()
    try:
        df = leer_datos(ruta)
    except Exception as exc:  # CSV corrupto o bloqueado: avisar sin romper la app
        st.warning(f"No se pudo leer el archivo de métricas ({ruta}): {exc}")
        return
    if df.empty:
        st.info(
            f"Aún no hay métricas registradas en `{ruta}`. Analiza una imagen en la app "
            "principal y aparecerán aquí (el calentamiento no se cuenta)."
        )
        return

    con_gpu = hay_gpu(df)
    _tarjetas(st, df, hay_gpu(df.tail(1)))
    if not con_gpu:
        st.caption("Sin GPU NVIDIA: se muestran solo métricas de CPU.")

    st.markdown(f"**Análisis registrados:** {len(df)}")
    c1, c2 = st.columns(2)
    with c1:
        mostrar_figura(st, _grafica_etapas(go, df), f"{clave}_etapas")
    if "ms_total" in df.columns and df["ms_total"].notna().any():
        p50 = float(df["ms_total"].quantile(0.50))
        n_total = int(df["ms_total"].notna().sum())
        p95 = float(df["ms_total"].quantile(0.95)) if n_total >= MIN_P95 else None
        with c2:
            mostrar_figura(st, _grafica_total(go, df, p50, p95), f"{clave}_total")
    if "cpu_pct_prom" in df.columns:
        mostrar_figura(st, _grafica_recursos(go, df, con_gpu), f"{clave}_recursos")

    tabla = _tabla_percentiles(df)
    if not tabla.empty:
        st.markdown(f"**Percentiles** (p95 solo con ≥{MIN_P95} análisis)")
        st.dataframe(tabla, hide_index=True)
        st.caption(
            "GPU % y VRAM (pico) son del dispositivo completo; CPU % y RAM son del "
            "proceso que analiza."
        )
