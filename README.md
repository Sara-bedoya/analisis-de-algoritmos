# Análisis de Algoritmos en el contexto de la Bibliometría

Proyecto final de la asignatura Análisis de Algoritmos.

## Descripción

Aplicación para el análisis computacional de abstracts de 20 artículos científicos relacionados con Inteligencia Artificial Generativa.

## Estado del proyecto

Implementación local de los requerimientos 1–3; despliegue externo y documento
final de entrega pendientes de definición y cierre.

El diseño y la arquitectura están descritos en
[`docs/diseno_tecnico.md`](docs/diseno_tecnico.md).

## Requerimiento 1: similitud textual

La comparación está organizada en `src/similarity`: carga del corpus, medidas
clásicas, adaptadores de embeddings, servicio de comparación e interfaz CLI.
Los algoritmos clásicos están implementados en el proyecto. Los modelos
preentrenados no se incluyen en el repositorio.

Para comparar los abstracts de los artículos 1 y 2 con las cuatro medidas
clásicas:

```powershell
python main.py --ids 1 2
```

Para comparar con todos los métodos, se requieren además archivos locales de
GloVe y FastText en formato de vectores de texto:

```powershell
python main.py --ids 1 2 --glove data/models/glove-wiki-gigaword-50.gz --fasttext data/models/fasttext-wiki-news-subwords-300.gz
```

`--salida ruta.csv` exporta los resultados. La documentación del módulo está
en `docs/similitud_textual.md`; las fuentes y condiciones de los modelos están
en `docs/modelos_embeddings.md`. El cálculo detallado para los artículos 1 y 10
está en `docs/caso_estudio_similitud.md`.

## Requerimiento 2: clustering jerárquico

El clustering usa distancia coseno sobre TF-IDF y construye desde cero los
árboles *single*, *complete* y *average linkage*. Ejecuta:

```powershell
python main.py clustering
```

El comando guarda los tres dendrogramas, las fusiones, la matriz de distancias,
las puntuaciones silhouette y las asignaciones del mejor corte en
`output/clustering/`. La metodología está en `docs/clustering_jerarquico.md`.

## Requerimiento 3: búsqueda de cadenas

La búsqueda exacta se ejecuta sobre los abstracts originales, distingue
mayúsculas y conserva puntuación y saltos de línea. Para buscar `AI`:

```powershell
python main.py busqueda --patron "AI" --repeticiones 7
```

Las posiciones y las medianas de siete mediciones separadas de
preprocesamiento/búsqueda se guardan en `output/searching/`. La implementación y el caso de estudio están
documentados en `docs/busqueda_cadenas.md`.

## Top 15 de palabras (ordenamiento)

Counting Sort, Radix Sort y Bucket Sort se ejecutan sobre las tuplas
`(frecuencia, palabra)` y muestran el Top 15 desde el mismo punto de entrada:

```powershell
python main.py top15
```

También acepta `python main.py sorting` y `python main.py ordenamiento`.

## Diseño y entrega

La matriz de cumplimiento está en `docs/revision_cumplimiento.md`; arquitectura
y declaración de apoyo de IA, en `docs/diseno_tecnico.md`. Los módulos se
ejecutan desde la raíz del proyecto; el PDF no fija una plataforma concreta
para el despliegue.

## Interfaz web

La interfaz Streamlit vive en este mismo proyecto y reutiliza los módulos
existentes. Para iniciarla localmente:

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

La guía de publicación en Streamlit Community Cloud y sus límites está en
[`docs/despliegue_streamlit.md`](docs/despliegue_streamlit.md). La app ofrece
similitud, clustering, búsqueda y Top 15; los embeddings se activan solo cuando sus
archivos locales están presentes.
