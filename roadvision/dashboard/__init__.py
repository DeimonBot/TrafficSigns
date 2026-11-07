"""Dashboard visual de métricas de uso del computador (spec §5.5 y §5.6).

- `panel.render_panel`: panel reutilizable con el histórico del CSV de métricas.
- `app_metrics.py`: app Streamlit independiente con monitor del sistema en vivo.

Los imports de streamlit y plotly son perezosos: importar este paquete no los carga.
"""
