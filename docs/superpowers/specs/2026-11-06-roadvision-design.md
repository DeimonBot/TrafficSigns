# RoadVision — Diseño (versión sin entrenamiento)

- **Fecha:** 2026-11-06
- **Estado:** borrador para aprobación del equipo
- **Repo:** https://github.com/DeimonBot/TrafficSigns

---

## 1. Intención

RoadVision es una aplicación de escritorio (interfaz web local con Streamlit) donde el usuario carga una fotografía de una vía y la aplicación:

1. detecta las señales de tránsito que aparecen,
2. clasifica cada una,
3. muestra su significado,
4. presenta un resumen de lo encontrado.

La aplicación corre en un PC con GPU NVIDIA y, mientras procesa, **muestra en la terminal las métricas de consumo y rendimiento del modelo** (tiempos por etapa, uso de GPU, VRAM, temperatura, potencia, CPU, RAM). Esa parte es tan importante como el reconocimiento: el proyecto debe poder demostrar cómo se comportan los modelos dentro del equipo.

El contexto es Colombia: las fotos serán de calles colombianas, cuya señalización sigue el *Manual de Señalización Vial de Colombia* (Res. 1885 de 2015). **No se entrena ni se ajusta ningún modelo**: se usan modelos preentrenados públicos de Hugging Face, combinados con lógica propia.

## 2. Metas, no-metas y anti-metas

### Metas
- Cargar una foto (JPG/PNG) desde la interfaz y obtener el análisis completo en una sola acción.
- Reconocer bien las señales **reglamentarias** colombianas que coinciden con el estándar europeo: PARE, CEDA EL PASO, velocidad máxima (círculo con borde rojo y número), prohibido adelantar, prohibido el paso, etc.
- Para las señales **preventivas** colombianas (rombo amarillo), que ningún modelo preentrenado conoce, dar al menos la **categoría** ("Señal preventiva") y, cuando sea posible, una identificación aproximada marcada como tal.
- Mostrar significado en español y un resumen (conteo por señal y por categoría).
- Ejecutar los modelos en GPU NVIDIA (CUDA) y caer a CPU sin romperse si no hay GPU.
- Imprimir en terminal métricas fiables por cada análisis y guardarlas en CSV.
- Poder repetir la instalación en el PC de la GPU siguiendo el README, paso a paso.

### No-metas
- Entrenar o afinar modelos (queda explícitamente fuera; ver sección 11).
- Video, webcam o tiempo real.
- Cubrir el catálogo completo de señales colombianas (son más de cien); se cubre un subconjunto común.
- Despliegue en la nube, multiusuario, cuentas, base de datos.
- App móvil.

### Anti-metas (si pasa esto, el proyecto falla aunque "funcione")
- **Etiquetar con seguridad una señal equivocada.** Siempre es preferible "Señal no reconocida" a un nombre incorrecto con 95 % de confianza.
- Colores invertidos en la imagen (rojo que se ve azul).
- Métricas engañosas: tiempos medidos sin sincronizar la GPU, o que incluyen la primera inferencia de calentamiento sin decirlo.
- Que la app solo funcione en el PC de quien la programó.

## 3. Restricciones

- **Hardware de ejecución:** un PC con Windows y GPU NVIDIA (modelo exacto aún no confirmado). No es el PC donde se escribió este documento.
- **Hardware de desarrollo:** los integrantes pueden programar en PCs sin NVIDIA; todo debe correr en CPU (más lento) para desarrollar y probar.
- **Python** 3.11 o 3.12.
- **PyTorch con CUDA:** en Windows, `pip install torch` instala por defecto la versión solo-CPU. La instalación debe usar el índice de pytorch.org con la variante CUDA correspondiente al driver del PC.
- **Pesos de modelos:** se descargan del Hub, nunca se suben al repo (`models/`, `*.pt`, `*.safetensors`, `*.bin` en `.gitignore`).
- **Licencias:** todos los modelos del camino principal son Apache-2.0 o MIT. No se usa Ultralytics en el camino principal (evita la licencia AGPL-3.0).

## 4. Decisiones

### Tomadas por el equipo
- Interfaz con **Streamlit**.
- Ejecución en PC con **GPU NVIDIA**.
- Métricas de consumo **en la terminal**.
- **Sin entrenamiento**.
- Funcionalidades: cargar foto, detectar, clasificar, mostrar significado, resumen.

### Tomadas por mí (revisar)
- Arquitectura en etapas: detector genérico + clasificador GTSRB + respaldo CLIP.
- Modelos concretos (sección 5.2).
- Catálogo de señales en un archivo YAML, con tabla de equivalencias GTSRB → Colombia.
- Conjunto de prueba con etiquetas por foto (no por caja).
- Panel de rendimiento opcional dentro de la interfaz, además de la terminal.
- Script de benchmark y comando de línea (`cli`) además de la interfaz.
- Verificación del número en los límites de velocidad con CLIP (5.3).
- Criterio de éxito (sección 6): cero errores "reconocida" en el conjunto de prueba manda sobre el recall; las "aproximadas" equivocadas se toleran porque van marcadas, pero se reportan.
- Se asume acceso intermitente al PC NVIDIA: todo lo que solo se puede probar allí queda en scripts y en una lista de verificación que cualquiera del equipo puede correr.
- Los contratos compartidos (`Resultado` y las interfaces `Detector`, `Clasificador`, `Respaldo`) se fijan al inicio de H1 para poder trabajar en paralelo.

## 5. Arquitectura

### 5.1 Flujo de una foto

```
Foto subida
  → normalizar: corregir rotación EXIF, convertir a RGB
  → DETECTOR (OWLv2, consulta "traffic sign"): cajas + puntaje
  → filtrar cajas: puntaje mínimo, tamaño mínimo, supresión de duplicados
  → recortar cada caja con margen
  → CLASIFICADOR GTSRB (ViT), todos los recortes en un solo lote
  → DECISIÓN por recorte (5.3):
        reconocida por GTSRB  → entrada del catálogo
        si no                 → RESPALDO CLIP → señal aproximada / solo categoría / no reconocida
  → RESULTADO: lista de señales + resumen + métricas
  → interfaz dibuja; terminal imprime métricas; CSV guarda métricas
```

El núcleo (`roadvision/`) no sabe nada de Streamlit. La función central es:

```python
analizar(imagen: PIL.Image) -> Resultado
```

`Resultado` es una estructura de datos con: la lista de señales (caja, recorte, id de catálogo o `None`, nombre en español, categoría, significado, confianza, fuente `gtsrb|clip|ninguna`, si es aproximada), el resumen (conteos) y las métricas del análisis.

### 5.2 Modelos

| Rol | Modelo (Hugging Face) | Notas |
|---|---|---|
| Detector | `google/owlv2-base-patch16-ensemble` | Detección por texto (zero-shot). Se usa como detector **genérico**: solo responde "aquí hay una señal". Encuentra señales de cualquier país/forma. Consultas iniciales: `"a traffic sign"`, `"a road sign"`. |
| Clasificador | `kelvinandreas/vit-traffic-sign-GTSRB` | ViT, 43 clases GTSRB, entrada 224×224, etiquetas con nombre (`"Stop"`, `"Speed limit (50km/h)"`…). |
| Respaldo | `openai/clip-vit-base-patch32` | Clasificación zero-shot por texto para lo que GTSRB no cubre. Las descripciones van **en inglés** (el modelo se entrenó en inglés); el usuario ve el resultado en español. |

Los tres se cargan con `transformers`. Cada rol está detrás de una interfaz simple (`Detector`, `Clasificador`, `Respaldo`) para poder cambiar el modelo desde la configuración. Alternativas previstas, no obligatorias:
- Detector: `ankitjha07/Traffic-Sign-Detection-YOLOv8` (usado como detector genérico, ignorando su clase) — requiere Ultralytics; solo para comparación.
- Respaldo: `openai/clip-vit-large-patch14` si la GPU tiene VRAM suficiente y mejora el conjunto de prueba.

Dispositivo: CUDA si está disponible, si no CPU. En CUDA se usa fp16 por defecto (configurable); en CPU, fp32.

### 5.3 Lógica de decisión por recorte

Esta es la pieza que evita la anti-meta principal.

1. **GTSRB.** Se toma la clase con mayor probabilidad.
   - Si la probabilidad ≥ `umbral_gtsrb` **y** esa clase tiene equivalente colombiano en el catálogo → **reconocida** (fuente `gtsrb`).
   - Si la clase **no aplica en Colombia** → se ignora el resultado de GTSRB y se pasa a CLIP. Caso crítico: la clase GTSRB *"Priority road"* es un **rombo amarillo**; las preventivas colombianas también son rombos amarillos, así que el ViT tenderá a llamar *"Priority road"* a cualquier preventiva colombiana con mucha confianza. Lo mismo para las clases "End of …" alemanas.
   - Si la probabilidad < `umbral_gtsrb` → pasa a CLIP.
   - **Verificación de límites de velocidad.** GTSRB solo conoce 20/30/50/60/70/80/100/120 km/h; en Colombia también hay, por ejemplo, 40 y 90, y el ViT llamaría "30" o "50" a un "40" con mucha confianza. Por eso, cuando la clase GTSRB es un límite de velocidad, CLIP compara el recorte contra descripciones del tipo `"a circular speed limit road sign showing the number N"` para cada valor del catálogo (incluidos los que GTSRB no tiene):
     - si el número de CLIP coincide con el del ViT → **reconocida** (fuente `gtsrb`);
     - si CLIP da con confianza ≥ `umbral_clip_senal` un número que GTSRB no tiene → **reconocida aproximada** con ese número (fuente `clip`);
     - en cualquier otro caso → **"Velocidad máxima — valor no verificado"** (categoría reglamentaria, sin número).
2. **CLIP (respaldo).** Se compara el recorte contra las descripciones en inglés de las entradas del catálogo que tienen `clip` definido, más una descripción por categoría.
   - Si la mejor señal específica supera `umbral_clip_senal` → **reconocida aproximada** (fuente `clip`, se muestra como "aproximada").
   - Si no, pero la mejor categoría supera `umbral_clip_categoria` → **solo categoría** ("Señal preventiva — tipo no identificado").
   - Si no → **no reconocida**.

Valores iniciales (se ajustan con el conjunto de prueba, sección 6): `umbral_det` 0.15, `umbral_gtsrb` 0.80, `umbral_clip_senal` 0.50, `umbral_clip_categoria` 0.50, `margen_recorte` 10 %, `tamano_minimo_caja` 20 px. Todos viven en `config.py` y los de detección/clasificación son ajustables desde la barra lateral.

### 5.4 Catálogo de señales (`data/catalogo_senales.yaml`)

Un archivo editable sin programar. Cada entrada:

```yaml
- id: pare
  nombre: "PARE"
  categoria: reglamentaria        # reglamentaria | preventiva | informativa
  significado: "Detenga completamente el vehículo y ceda el paso antes de continuar."
  gtsrb: ["Stop"]                 # etiquetas GTSRB equivalentes (puede ser vacío)
  clip: "a red octagonal stop sign"   # descripción en inglés para CLIP (opcional)
```

Además, una lista `gtsrb_sin_equivalente` con las etiquetas GTSRB que no aplican en Colombia (al menos *Priority road*, *End of priority road* si existiera, *End of speed limit (80km/h)*, *End of all speed and passing limits*, *End of no passing*, *End of no passing veh > 3.5 tons*). El constructor la completa revisando las 43 clases.

Contenido inicial esperado (unas 20–30 entradas):
- **Reglamentarias (vía GTSRB):** PARE, CEDA EL PASO, velocidades máximas 20/30/50/60/70/80/100/120, prohibido adelantar, prohibido el paso / no entre, y las que el constructor encuentre equivalentes en el Manual.
- **Velocidades sin equivalente GTSRB (vía CLIP):** al menos 40 y 90 km/h (el constructor confirma con el Manual qué valores se usan en Colombia), más una entrada genérica "Velocidad máxima — valor no verificado".
- **Reglamentarias (vía CLIP):** prohibido girar a la izquierda/derecha, prohibido estacionar — no existen en GTSRB.
- **Preventivas (vía CLIP, y vía GTSRB cuando el símbolo coincide):** resalto / reductor de velocidad, zona escolar, peatones, semáforo adelante, curva a la izquierda/derecha, intersección, glorieta, vía resbaladiza, trabajos en la vía.
- **Categorías genéricas para CLIP:** "yellow diamond-shaped warning road sign" (preventiva), "white circular road sign with red border" (reglamentaria), "blue rectangular informational road sign" (informativa).

Los significados se redactan con base en el Manual de Señalización Vial de Colombia 2015.

**Regla verificable:** cada una de las 43 etiquetas del clasificador debe estar mapeada a una entrada o listada en `gtsrb_sin_equivalente`. Una prueba automática lo comprueba.

### 5.5 Interfaz (Streamlit, en español)

- **Barra lateral:** umbral de detección, umbral de clasificación, casilla "mostrar no reconocidas", dispositivo en uso (GPU y su nombre, o CPU), casilla "mostrar panel de rendimiento".
- **Zona principal:**
  1. Cargador de archivo (JPG, PNG).
  2. Imagen con las cajas dibujadas, numeradas y coloreadas por categoría (rojo reglamentaria, amarillo preventiva, azul informativa, gris no reconocida). Colores RGB correctos.
  3. Tabla: número, miniatura del recorte, señal, categoría, significado, confianza, fuente (y la marca "aproximada" cuando aplique).
  4. Resumen: total de señales, conteo por categoría, lista de señales distintas con su significado.
  5. (Opcional) Panel plegable "Rendimiento" con las mismas métricas que la terminal.
- Los modelos se cargan una sola vez (`@st.cache_resource`); se muestra un indicador mientras analiza.
- Si la foto no tiene señales, se dice explícitamente.

### 5.6 Métricas (`roadvision/metrics.py`)

**Al iniciar la app** (una vez) se imprime: GPU (nombre, VRAM total, driver, versión CUDA), versión de PyTorch, dispositivo y precisión elegidos, tiempo de carga y VRAM ocupada por cada modelo, y el resultado de una inferencia de calentamiento (marcada como tal, excluida de las estadísticas).

**Por cada análisis** se imprime una tabla (con `rich`) y se agrega una fila a `logs/metricas.csv`:

| Métrica | Fuente |
|---|---|
| Tiempo por etapa: preproceso, detección, clasificación, respaldo CLIP, total | `time.perf_counter()` con `torch.cuda.synchronize()` antes de cada lectura cuando se usa CUDA |
| Imágenes por segundo equivalentes | 1 / total |
| Uso de GPU % (pico y promedio durante el análisis) | `nvidia-ml-py`, muestreado cada ~100 ms por un hilo aparte **mientras** corre el análisis |
| VRAM usada (total del dispositivo) | `nvidia-ml-py` |
| VRAM pico de PyTorch en el análisis | `torch.cuda.reset_peak_memory_stats()` / `max_memory_allocated()` |
| Temperatura y potencia (W) | `nvidia-ml-py` |
| CPU % y RAM del proceso | `psutil` |
| Tamaño de la imagen, número de detecciones, reconocidas, aproximadas, no reconocidas | pipeline |

Sin GPU NVIDIA, las métricas de GPU se omiten (se indica "CPU") sin errores.

**Benchmark** (`scripts/benchmark.py`): recorre una carpeta de fotos, repite N veces, excluye el calentamiento y reporta promedio y p95 por etapa. Permite comparar `--device cuda|cpu` y `--dtype fp16|fp32`. Sirve para la sección de consumo del informe.

### 5.7 Estructura del repo

```
TrafficSigns/
├─ app.py                        # solo interfaz Streamlit
├─ roadvision/
│  ├─ __init__.py
│  ├─ config.py                  # ids de modelos, umbrales, dispositivo, precisión
│  ├─ device.py                  # elección CUDA/CPU, fp16, calentamiento
│  ├─ imaging.py                 # EXIF, RGB, recortes con margen, dibujo de cajas
│  ├─ detector.py                # interfaz Detector + OWLv2
│  ├─ classifier.py              # interfaz Clasificador + ViT GTSRB
│  ├─ fallback.py                # respaldo CLIP
│  ├─ catalog.py                 # carga y valida el YAML
│  ├─ pipeline.py                # analizar(imagen) -> Resultado
│  ├─ metrics.py                 # medidor por etapa, muestreo GPU, reporte rich, CSV
│  └─ cli.py                     # python -m roadvision.cli foto.jpg (sin interfaz)
├─ data/
│  └─ catalogo_senales.yaml
├─ samples/                      # fotos de prueba propias (pocas, livianas)
├─ eval/
│  └─ etiquetas.csv              # conjunto de prueba (sección 6)
├─ scripts/
│  ├─ download_models.py         # descarga los 3 modelos a la caché local
│  ├─ check_gpu.py               # verifica CUDA, GPU y nvidia-ml-py
│  ├─ benchmark.py
│  └─ evaluate.py                # mide aciertos contra eval/etiquetas.csv
├─ tests/
├─ logs/                         # ignorado por git
├─ requirements.txt              # todo excepto torch
└─ README.md                     # instalación paso a paso (incluye torch con CUDA)
```

`.gitignore` debe añadir `*.pt` y `logs/`.

### 5.8 Dependencias

`torch` + `torchvision` (instalados aparte con el índice CUDA), `transformers`, `pillow`, `streamlit`, `nvidia-ml-py`, `psutil`, `rich`, `pyyaml`, `pytest`. Versiones fijadas en `requirements.txt`.

## 6. Evaluación (sin entrenar, pero midiendo)

Sin entrenamiento, los umbrales y la elección de modelos se deciden con datos propios:

- **Conjunto de prueba:** 30–50 fotos de calles colombianas tomadas por el equipo, variadas (día/noche, cerca/lejos, distintas zonas). Se guardan fuera del repo si pesan mucho; en el repo, solo unas pocas en `samples/`.
- **Etiquetado simple:** `eval/etiquetas.csv` con una fila por foto y la lista de ids de catálogo que aparecen (no se dibujan cajas).
- **`scripts/evaluate.py`** reporta por foto y en total: señales esperadas encontradas (recall), señales reportadas que no estaban (falsos positivos), aproximadas correctas/incorrectas, no reconocidas.
- Con esto se ajustan los umbrales de 5.3 y se decide si vale la pena CLIP *large* o comparar con el detector YOLO.
- **Criterio de éxito y orden de ajuste:**
  1. Primero, subir umbrales hasta que **ninguna** señal salga como "reconocida" con un nombre equivocado en el conjunto de prueba (como máximo una, explicada en el informe).
  2. Después, con esa restricción, maximizar cuántas señales esperadas se encuentran.
  3. Las "aproximadas" equivocadas se toleran porque se muestran como tales, pero su tasa se reporta.
  - No hay meta numérica de recall; se reporta por categoría (reglamentarias vs. preventivas).
- Los resultados se presentan honestamente en el informe, separando reglamentarias y preventivas.

## 7. Hitos

| Hito | Contenido | Depende de |
|---|---|---|
| H0 Base | Estructura del repo, `.gitignore`, requirements, README de instalación, `check_gpu.py`, `download_models.py`, y una **lista de verificación para el PC NVIDIA** (instalar, correr `check_gpu.py`, correr `benchmark.py` en fp16 y fp32, comparar resultados). Instalación verificada en el PC NVIDIA. **Empezar a tomar las fotos de prueba.** | — |
| H1 Núcleo | **Primero, fijar los contratos** (`Resultado`, interfaces `Detector`/`Clasificador`/`Respaldo`). Luego `imaging`, `detector`, `classifier`, `pipeline` básico (sin respaldo), `cli` que imprime las detecciones. | H0 |
| H2 Catálogo y decisión | YAML con equivalencias, `catalog`, lógica 5.3 completa con respaldo CLIP, prueba de cobertura de las 43 etiquetas. | H1 |
| H3 Interfaz | `app.py` completo según 5.5. | H2 |
| H4 Métricas | `metrics` integrado en pipeline, reporte de arranque, CSV, `benchmark.py`. | H1 (en paralelo a H2/H3) |
| H5 Evaluación | `etiquetas.csv`, `evaluate.py`, ajuste de umbrales, tabla de resultados para el informe. | H2 + fotos |

Pruebas automáticas mínimas: cobertura del catálogo, pipeline sobre una imagen de `samples/` en CPU, `metrics` funcionando sin GPU, normalización EXIF/RGB.

## 8. Riesgos

| Riesgo | Mitigación |
|---|---|
| OWLv2 reduce la imagen a 960 px de lado: señales lejanas en fotos de 4000 px pueden perderse. | Documentar que la app apunta a fotos donde la señal es razonablemente visible. Opción futura: análisis por mosaicos (no incluido). |
| El ViT GTSRB es sobreconfiado con señales que no conoce. | Umbral alto + lista `gtsrb_sin_equivalente` + respaldo CLIP. |
| CLIP es débil distinguiendo símbolos finos (curva vs. resalto). | Se muestra como "aproximada"; si no es claro, solo categoría. |
| CLIP puede no leer bien los números de los límites de velocidad. | Falla segura: la señal queda como "Velocidad máxima — valor no verificado". Si el conjunto de prueba muestra que pasa mucho, se evalúa un OCR (fuera de este diseño). |
| Instalación de PyTorch con CUDA en Windows. | README paso a paso + `check_gpu.py`. |
| Primera ejecución sin internet. | `download_models.py` se corre durante la instalación. |
| Falsos positivos del detector (avisos comerciales, vallas). | Umbral de detección + el clasificador/respaldo los deja como "no reconocida" (ocultables con la casilla). |

## 9. Lo que queda al constructor

- Versiones exactas de librerías.
- Consultas de texto exactas para OWLv2 y descripciones exactas para CLIP (iterando con el conjunto de prueba).
- Supresión de duplicados entre consultas del detector.
- Estilo visual de la interfaz dentro de lo descrito en 5.5.
- Formato exacto de la tabla de terminal y columnas del CSV (mínimo las de 5.6).
- Completar la tabla de equivalencias GTSRB → Colombia con el Manual.

## 10. Abierto (lo decide el equipo)

- Fecha de entrega (define si entran los extras: panel de rendimiento en la interfaz, CLIP large, comparación con YOLO).
- Cuántos integrantes son y quién toma qué hito.
- Modelo y VRAM de la GPU del PC de ejecución (afecta fp16 por defecto y si cabe CLIP large; la configuración permite cambiarlo).

## 11. Evolución futura (fuera de este diseño)

Si más adelante se quiere precisión real con señales colombianas: etiquetar fotos propias y ajustar un YOLO (p. ej. YOLO11s) que detecte y clasifique en un paso. Gracias a las interfaces de 5.2, entraría como un nuevo `Detector` sin cambiar la interfaz ni las métricas.
