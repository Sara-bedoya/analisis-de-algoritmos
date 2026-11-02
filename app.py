"""Interfaz web Streamlit para los requerimientos 1, 2, 3 y el Top 15."""

from __future__ import annotations

import csv
import io
import os
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from src.clustering.hierarchical import METODOS, cortar_arbol
from src.clustering.service import evaluar_enlaces, matriz_distancias_tfidf
from src.clustering.visualization import guardar_dendrograma
from src.searching.service import comparar_patron
from src.similarity.corpus import cargar_corpus
from src.similarity.embeddings import cargar_fasttext, cargar_glove
from src.similarity.service import comparar_articulos
from src.sorting.service import pares_desde_articulos, ranking_por_algoritmo


ROOT = Path(__file__).resolve().parent
CORPUS_PATH = ROOT / "data" / "corpus_articulos.csv"
EMBEDDINGS_DIR = Path(os.environ.get("EMBEDDINGS_DIR", ROOT / "data" / "models"))


def _resolver_ruta_modelo(nombre: str) -> Path:
    """Busca primero el modelo compacto de corpus; si no, el modelo grande completo."""
    if nombre == "GloVe":
        compacto = EMBEDDINGS_DIR / "glove_corpus.txt.gz"
        completo = EMBEDDINGS_DIR / "glove-wiki-gigaword-50.gz"
    else:
        compacto = EMBEDDINGS_DIR / "fasttext_corpus.txt.gz"
        completo = EMBEDDINGS_DIR / "fasttext-wiki-news-subwords-300.gz"
    return compacto if compacto.is_file() else completo


MODEL_PATHS = {
    "GloVe": _resolver_ruta_modelo("GloVe"),
    "FastText": _resolver_ruta_modelo("FastText"),
}

ACCENTOS = {
    "lavanda": "#6656D9",
    "menta": "#168C82",
    "coral": "#E57C5B",
    "ambar": "#C9842A",
}

st.set_page_config(
    page_title="Análisis de abstracts científicos",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      [data-testid="stAppViewContainer"] {
        background:
          radial-gradient(ellipse at 90% 0%, rgba(102,86,217,.09), transparent 32%),
          #F7F6FC;
      }
      [data-testid="stHeader"] { background: transparent; }
      [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #272344 0%, #342F5D 100%);
      }
      [data-testid="stSidebar"] * { color: #F7F5FF; }
      [data-testid="stSidebar"] [data-testid="stMetricValue"] { color: #BEB5FF; }
      .block-container { max-width: 1420px; padding-top: 2.4rem; }
      .hero {
        position: relative; overflow: hidden; padding: 2rem 2.2rem;
        border-radius: 24px; color: #fff;
        background: linear-gradient(118deg, #292448 0%, #5145A3 64%, #7668DD 100%);
        box-shadow: 0 18px 45px rgba(48, 40, 101, .18);
        margin-bottom: 1.2rem;
      }
      .hero:after {
        content: ""; position: absolute; width: 230px; height: 230px;
        border-radius: 50%; right: -58px; top: -112px;
        background: rgba(255,255,255,.10);
        box-shadow: -82px 178px 0 -42px rgba(255,255,255,.08);
      }
      .hero-kicker {
        color: #CFC9FF; font-size: .76rem; font-weight: 750;
        letter-spacing: .13em; text-transform: uppercase;
      }
      .hero h1 { color: #fff; font-size: clamp(2rem, 4vw, 3rem); margin: .55rem 0 .45rem; }
      .hero p { color: #E8E5FF; font-size: 1.05rem; max-width: 780px; margin: 0; }
      .hero-chip {
        display: inline-block; margin-top: 1.2rem; margin-right: .45rem;
        padding: .35rem .7rem; border: 1px solid rgba(255,255,255,.25);
        border-radius: 999px; color: #F4F2FF; font-size: .78rem;
        background: rgba(255,255,255,.09);
      }
      .feature-card {
        min-height: 145px; padding: 1.1rem 1.2rem; border-radius: 18px;
        background: #fff; border: 1px solid #E8E5F3;
        box-shadow: 0 8px 22px rgba(41,36,72,.045);
      }
      .feature-number { font-size: .73rem; font-weight: 800; letter-spacing: .1em; }
      .feature-card h3 { margin: .35rem 0 .3rem; font-size: 1.1rem; color: #25233A; }
      .feature-card p { margin: 0; color: #6D6A80; font-size: .88rem; line-height: 1.45; }
      .section-heading { margin: .2rem 0 1.1rem; }
      .section-heading .eyebrow {
        font-size: .73rem; font-weight: 800; text-transform: uppercase;
        letter-spacing: .12em; color: #6656D9;
      }
      .section-heading h2 { margin: .22rem 0 .3rem; color: #25233A; font-size: 1.8rem; }
      .section-heading p { color: #706D82; margin: 0; }
      .stTabs [data-baseweb="tab-list"] { gap: .5rem; border-bottom: 1px solid #E3E0EF; }
      .stTabs [data-baseweb="tab"] {
        height: 3.1rem; padding: 0 1.1rem; border-radius: 12px 12px 0 0;
        color: #6D6982; font-weight: 650;
      }
      .stTabs [aria-selected="true"] { color: #5145A3 !important; }
      div.stButton > button[kind="primary"] {
        border: 0; border-radius: 11px; font-weight: 700;
        background: linear-gradient(100deg, #6656D9, #8273E8);
        box-shadow: 0 6px 16px rgba(102,86,217,.2);
      }
      div.stButton > button[kind="primary"]:hover { background: #5145A3; }
      [data-testid="stMetric"] {
        background: #fff; border: 1px solid #E8E5F3; padding: 1rem 1.1rem;
        border-radius: 16px; box-shadow: 0 5px 16px rgba(41,36,72,.04);
      }
      [data-testid="stMetricLabel"] { color: #6D6982; }
      [data-testid="stMetricValue"] { color: #342F5D; }
      [data-testid="stDataFrame"] { border: 1px solid #E8E5F3; border-radius: 14px; }
      hr { border-color: #E6E3F0; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def obtener_corpus() -> dict[int, dict[str, str]]:
    """Carga el catálogo una vez y lo conserva en la caché de datos."""
    return cargar_corpus(CORPUS_PATH)


@st.cache_resource(show_spinner="Leyendo vectores del modelo; esto puede tardar.")
def obtener_embeddings(nombre: str, ruta: str, vocabulario: tuple[str, ...]):
    """Lee una vez el modelo elegido y conserva solo los vectores del corpus."""
    if nombre == "GloVe":
        return cargar_glove(ruta, set(vocabulario))
    return cargar_fasttext(ruta, set(vocabulario))


@st.cache_data(show_spinner=False)
def obtener_pares_frecuencia(textos: tuple[str, ...]) -> list[tuple[int, str]]:
    """Cuenta términos de contenido a partir de los abstracts del corpus."""
    articulos_minimos = {indice: {"abstract": texto} for indice, texto in enumerate(textos)}
    return pares_desde_articulos(articulos_minimos)


@st.cache_data(show_spinner=False)
def obtener_benchmark_embeddings() -> dict[tuple[int, int], dict[str, Any]]:
    """Carga resultados empíricos de embeddings previamente verificados para el caso de estudio."""
    ruta_benchmark = ROOT / "data" / "resultados_similitud_1_10.csv"
    if not ruta_benchmark.is_file():
        return {}
    benchmark = {}
    try:
        with ruta_benchmark.open("r", encoding="utf-8-sig") as f:
            lector = csv.DictReader(f)
            for fila in lector:
                id_a, id_b = int(fila["id_a"]), int(fila["id_b"])
                clave = (min(id_a, id_b), max(id_a, id_b))
                benchmark[clave] = {
                    "GloVe (coseno)": float(fila["GloVe (coseno)"]),
                    "FastText (coseno)": float(fila["FastText (coseno)"]),
                    "GloVe cobertura artículo A": fila.get("GloVe cobertura artículo A", ""),
                    "GloVe cobertura artículo B": fila.get("GloVe cobertura artículo B", ""),
                    "FastText cobertura artículo A": fila.get("FastText cobertura artículo A", ""),
                    "FastText cobertura artículo B": fila.get("FastText cobertura artículo B", ""),
                }
    except Exception:
        return {}
    return benchmark


def _etiqueta_articulo(identificador: int, articulos: dict[int, dict[str, str]]) -> str:
    return f"{identificador:02d} · {articulos[identificador]['titulo']}"


def _encabezado_seccion(etiqueta: str, titulo: str, descripcion: str) -> None:
    st.markdown(
        f"""
        <div class="section-heading">
          <div class="eyebrow">{etiqueta}</div>
          <h2>{titulo}</h2>
          <p>{descripcion}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _csv_descargable(registros: list[dict[str, Any]]) -> bytes:
    salida = io.StringIO()
    if registros:
        escritor = csv.DictWriter(salida, fieldnames=list(registros[0]))
        escritor.writeheader()
        escritor.writerows(registros)
    return salida.getvalue().encode("utf-8-sig")


def _init_estado() -> None:
    """Crea las claves de resultado para que un clic posterior no las borre."""
    for clave in ("similitud", "clustering", "busqueda", "frecuencias"):
        st.session_state.setdefault(clave, None)


def _render_similitud(articulos: dict[int, dict[str, str]]) -> None:
    _encabezado_seccion(
        "01 · Requerimiento 1",
        "Similitud textual",
        "Compara perspectivas léxicas y vectoriales entre artículos científicos.",
    )
    st.write(
        "Selecciona dos o más artículos. Se calculan Levenshtein normalizado, "
        "coseno TF-IDF, Jaccard y Dice."
    )
    ids = sorted(articulos)
    seleccion = st.multiselect(
        "Artículos del corpus",
        options=ids,
        default=ids[:2],
        format_func=lambda identificador: _etiqueta_articulo(identificador, articulos),
        key="similarity_ids",
    )

    modelos_disponibles = [nombre for nombre, ruta in MODEL_PATHS.items() if ruta.is_file()]
    benchmark_cache = obtener_benchmark_embeddings()
    modelos_elegidos: list[str] = []
    if modelos_disponibles:
        with st.expander("Modelos de embeddings (IA)", expanded=True):
            st.caption(
                "Modelos vectoriales densos disponibles (GloVe / FastText). "
                "Se calculan representaciones promedio y similitud coseno."
            )
            for nombre in modelos_disponibles:
                if st.checkbox(
                    f"Incluir {nombre}",
                    value=True,
                    key=f"include_{nombre}",
                ):
                    modelos_elegidos.append(nombre)
    else:
        st.info(
            "ℹ️ Modelos locales no detectados en data/models. "
            "Se calculan dinámicamente las 4 medidas clásicas en tiempo real. "
            "Para el par de artículos 1 y 10 (caso de estudio del proyecto), se integran además los "
            "resultados empíricos verificados de GloVe y FastText registrados en el corpus."
        )

    confirmacion = True
    if len(seleccion) > 6:
        st.warning(
            "Comparar muchos artículos puede tardar porque Levenshtein calcula "
            "distancias de edición para cada pareja. Confirma si quieres continuar."
        )
        confirmacion = st.checkbox(
            "Entiendo que la comparación puede tardar",
            key="confirm_large_comparison",
        )
    ejecutar = st.button("Calcular similitud", type="primary", key="run_similarity")
    if ejecutar:
        if len(seleccion) < 2:
            st.error("Selecciona al menos dos artículos.")
        elif not confirmacion:
            st.info("Confirma la comparación para continuar.")
        else:
            modelos = {}
            try:
                if modelos_elegidos:
                    vocabulario = tuple(sorted({
                        token
                        for articulo in articulos.values()
                        for token in articulo["abstract_normalizado"].split()
                    }))
                    for nombre in modelos_elegidos:
                        with st.spinner(f"Cargando {nombre}…"):
                            modelos[nombre] = obtener_embeddings(
                                nombre, str(MODEL_PATHS[nombre]), vocabulario
                            )
                with st.spinner("Calculando las medidas para los artículos seleccionados…"):
                    resultados = comparar_articulos(
                        articulos,
                        seleccion,
                        glove=modelos.get("GloVe"),
                        fasttext=modelos.get("FastText"),
                    )
            except (OSError, ValueError, RuntimeError) as error:
                st.error(f"No fue posible calcular la similitud: {error}")
            else:
                filas = []
                for resultado in resultados:
                    id_a = resultado["id_a"]
                    id_b = resultado["id_b"]
                    clave = (min(id_a, id_b), max(id_a, id_b))
                    medidas_finales = dict(resultado["medidas"])
                    coberturas_finales = dict(resultado["coberturas"])

                    if "GloVe (coseno)" not in medidas_finales and clave in benchmark_cache:
                        medidas_finales["GloVe (coseno)"] = benchmark_cache[clave]["GloVe (coseno)"]
                        medidas_finales["FastText (coseno)"] = benchmark_cache[clave]["FastText (coseno)"]
                        coberturas_finales["GloVe cobertura artículo A"] = benchmark_cache[clave]["GloVe cobertura artículo A"]
                        coberturas_finales["GloVe cobertura artículo B"] = benchmark_cache[clave]["GloVe cobertura artículo B"]
                        coberturas_finales["FastText cobertura artículo A"] = benchmark_cache[clave]["FastText cobertura artículo A"]
                        coberturas_finales["FastText cobertura artículo B"] = benchmark_cache[clave]["FastText cobertura artículo B"]

                    filas.append({
                        "ID A": id_a,
                        "Artículo A": articulos[id_a]["titulo"],
                        "ID B": id_b,
                        "Artículo B": articulos[id_b]["titulo"],
                        **medidas_finales,
                        **coberturas_finales,
                    })
                st.session_state.similitud = filas

    filas = st.session_state.similitud
    if not filas:
        return

    st.dataframe(filas, width="stretch", hide_index=True)
    st.metric("Parejas comparadas", len(filas))
    st.download_button(
        "Descargar comparación CSV",
        data=_csv_descargable(filas),
        file_name="resultados_similitud.csv",
        mime="text/csv",
    )
    st.caption(
        "Los métodos léxicos y los embeddings usan escalas distintas. "
        "Interpreta cada medida junto con su definición y cobertura."
    )


def _render_clustering(articulos: dict[int, dict[str, str]]) -> None:
    _encabezado_seccion(
        "02 · Requerimiento 2",
        "Clustering jerárquico",
        "Explora cómo se agrupan los abstracts y qué tan separados quedan los grupos.",
    )
    st.write(
        "Compara Single, Complete y Average Linkage sobre TF-IDF con distancia "
        "coseno. El mejor corte de cada árbol se elige mediante silhouette."
    )
    if st.button("Calcular clustering", type="primary", key="run_clustering"):
        with st.spinner("Construyendo los tres árboles y evaluando sus cortes…"):
            try:
                ids, distancias = matriz_distancias_tfidf(articulos)
                if not distancias:
                    raise ValueError("La matriz de distancias está vacía.")
                resultados = evaluar_enlaces(distancias)
            except (ValueError, RuntimeError) as error:
                st.error(str(error))
            else:
                st.session_state.clustering = {"ids": ids, "resultados": resultados}

    estado = st.session_state.clustering
    if not estado:
        return

    ids = estado["ids"]
    resultados = estado["resultados"]
    filas_resumen = [
        {
            "Enlace": metodo.title(),
            "Mejor k": resultados[metodo]["mejor_k"],
            "Silhouette": resultados[metodo]["mejor_silhouette"],
        }
        for metodo in METODOS
    ]
    metricas = st.columns(3)
    for columna, fila in zip(metricas, filas_resumen):
        columna.metric(
            f"{fila['Enlace']} · mejor silhouette",
            f"{fila['Silhouette']:.4f}",
            f"k = {fila['Mejor k']}",
        )
    st.dataframe(filas_resumen, width="stretch", hide_index=True)

    puntuaciones = [
        {"k": k, "Enlace": metodo.title(), "Silhouette": puntuacion}
        for metodo in METODOS
        for k, puntuacion in resultados[metodo]["puntuaciones"]
    ]
    st.markdown("#### Silhouette según el número de grupos")
    tabla_puntuaciones = pd.DataFrame(puntuaciones).pivot(
        index="k", columns="Enlace", values="Silhouette"
    )
    st.line_chart(tabla_puntuaciones)

    tabs = st.tabs([metodo.title() for metodo in METODOS])
    for metodo, tab in zip(METODOS, tabs):
        detalle = resultados[metodo]
        etiquetas = cortar_arbol(len(ids), detalle["fusiones"], detalle["mejor_k"])
        with tab:
            izquierda, derecha = st.columns([1.2, 1])
            with tempfile.TemporaryDirectory() as temp_dir:
                ruta = Path(temp_dir) / f"dendrograma_{metodo}.png"
                guardar_dendrograma(
                    articulos, ids, detalle["fusiones"], metodo, ruta
                )
                imagen = ruta.read_bytes()
            izquierda.image(imagen, width="stretch")
            grupos = [
                {
                    "Grupo": etiqueta + 1,
                    "ID": identificador,
                    "Artículo": articulos[identificador]["titulo"],
                }
                for identificador, etiqueta in zip(ids, etiquetas)
            ]
            derecha.markdown("**Artículos en el mejor corte**")
            derecha.dataframe(grupos, width="stretch", hide_index=True)
    st.caption(
        "Silhouette cercana a 0 indica separación débil. Un empate entre "
        "enlaces se reporta tal cual; no implica que exista un ganador único."
    )


def _render_busqueda(articulos: dict[int, dict[str, str]]) -> None:
    _encabezado_seccion(
        "03 · Requerimiento 3",
        "Búsqueda exacta",
        "Encuentra patrones en el texto original y compara tres algoritmos clásicos.",
    )
    st.write(
        "Escribe la cadena tal como debe aparecer. Se conserva mayúsculas, "
        "puntuación y espacios; las posiciones reportadas empiezan en 1."
    )
    patron = st.text_input("Patrón", placeholder="Ejemplo: AI", key="pattern")
    repeticiones = st.slider(
        "Repeticiones para calcular la mediana de tiempo",
        min_value=1,
        max_value=15,
        value=7,
        key="search_repetitions",
    )
    if st.button("Buscar patrón", type="primary", key="run_search"):
        if patron == "":
            st.error("Ingresa un patrón no vacío.")
        else:
            with st.spinner("Buscando en los 20 abstracts con los tres algoritmos…"):
                try:
                    resultados = comparar_patron(articulos, patron, repeticiones)
                except ValueError as error:
                    st.error(str(error))
                else:
                    st.session_state.busqueda = resultados

    resultados = st.session_state.busqueda
    if not resultados:
        return

    resumen = []
    ocurrencias = []
    for metodo, detalle in resultados.items():
        cantidad_total = 0
        for identificador, posiciones in detalle["ocurrencias"].items():
            cantidad_total += len(posiciones)
            if posiciones:
                ocurrencias.append({
                    "Algoritmo": metodo,
                    "ID": identificador,
                    "Artículo": articulos[identificador]["titulo"],
                    "Cantidad": len(posiciones),
                    "Posiciones (base 1)": "; ".join(
                        str(posicion + 1) for posicion in posiciones
                    ),
                })
        resumen.append({
            "Algoritmo": metodo,
            "Coincidencias": cantidad_total,
            "Preprocesamiento (ms)": detalle["tiempo_preparacion_ms"],
            "Búsqueda (ms)": detalle["tiempo_busqueda_ms"],
            "Comparaciones": detalle["comparaciones"],
            "Colisiones hash": detalle["colisiones"],
        })
    st.markdown("#### Rendimiento y coincidencias")
    metricas = st.columns(3)
    for columna, fila in zip(metricas, resumen):
        columna.metric(
            fila["Algoritmo"],
            f"{fila['Búsqueda (ms)']:.3f} ms",
            f"{fila['Coincidencias']} coincidencias",
        )
    st.dataframe(resumen, width="stretch", hide_index=True)
    if ocurrencias:
        st.dataframe(ocurrencias, width="stretch", hide_index=True)
    else:
        st.info("No se encontró el patrón en los abstracts del corpus.")
    st.download_button(
        "Descargar ocurrencias CSV",
        data=_csv_descargable(ocurrencias),
        file_name="ocurrencias_busqueda.csv",
        mime="text/csv",
    )
    st.caption(
        "Las comparaciones directas de Rabin–Karp cuentan las ventanas con "
        "hash coincidente que se verificaron carácter por carácter."
    )


def _render_frecuencias(articulos: dict[int, dict[str, str]]) -> None:
    _encabezado_seccion(
        "04 · Ordenamiento",
        "Top 15 de términos",
        "Counting, Radix y Bucket Sort ordenan tuplas (frecuencia, término).",
    )
    st.write(
        "Se tokenizan solo palabras alfabéticas y se eliminan stopwords en "
        "inglés. Cada algoritmo mueve la tupla completa; el ranking se recorta "
        "del arreglo ordenado, sin reconstruir un diccionario externo."
    )
    if st.button("Calcular Top 15", type="primary", key="run_top15"):
        with st.spinner("Contando términos y ordenando con los tres algoritmos…"):
            try:
                textos = tuple(
                    articulo.get("abstract") or articulo.get("abstract_normalizado", "")
                    for articulo in articulos.values()
                )
                pares = obtener_pares_frecuencia(textos)
                if not pares:
                    raise ValueError("No quedaron términos para ordenar.")
                rankings = ranking_por_algoritmo(pares, n=15)
            except (ValueError, TypeError, RuntimeError) as error:
                st.error(f"No fue posible calcular el Top 15: {error}")
            else:
                st.session_state.frecuencias = {
                    "vocabulario": len(pares),
                    "rankings": rankings,
                }

    estado = st.session_state.frecuencias
    if not estado:
        return

    st.metric("Términos distintos (sin stopwords)", estado["vocabulario"])
    rankings = estado["rankings"]
    coinciden = (
        rankings["Counting Sort"]
        == rankings["Radix Sort"]
        == rankings["Bucket Sort"]
    )
    if coinciden:
        st.success("Los tres algoritmos producen el mismo Top 15 a partir de las tuplas.")
    else:
        st.warning("Los rankings no coinciden; revisa la estabilidad de los algoritmos.")

    pestanas = st.tabs(list(rankings))
    filas_csv: list[dict[str, Any]] = []
    for pestana, (nombre, top_15) in zip(pestanas, rankings.items()):
        filas = [
            {"Posición": posicion, "Palabra": palabra, "Frecuencia": frecuencia}
            for posicion, (frecuencia, palabra) in enumerate(top_15, start=1)
        ]
        for fila in filas:
            filas_csv.append({"Algoritmo": nombre, **fila})
        with pestana:
            st.dataframe(filas, width="stretch", hide_index=True)

    st.download_button(
        "Descargar Top 15 CSV",
        data=_csv_descargable(filas_csv),
        file_name="top_15_algoritmos.csv",
        mime="text/csv",
    )
    st.caption(
        "El Top 15 son los últimos 15 elementos del arreglo ordenado de forma "
        "ascendente, invertidos. Palabras como and, the u of no aparecen porque "
        "se filtraron como stopwords."
    )


def main() -> None:
    """Punto de entrada de la interfaz."""
    _init_estado()
    st.markdown(
        """
        <div class="hero">
          <div class="hero-kicker">Proyecto · Análisis de algoritmos</div>
          <h1>Leer la ciencia con algoritmos</h1>
          <p>Explora similitud, agrupamiento, patrones de texto y el ranking
          de términos en un corpus de 20 artículos sobre IA en educación.</p>
          <span class="hero-chip">20 artículos</span>
          <span class="hero-chip">4 módulos interactivos</span>
          <span class="hero-chip">Algoritmos implementados desde cero</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not CORPUS_PATH.is_file():
        st.error(
            "No se encontró `data/corpus_articulos.csv`. Incluye el catálogo "
            "en el repositorio antes de iniciar o desplegar la aplicación."
        )
        st.stop()
    try:
        articulos = obtener_corpus()
    except (OSError, ValueError) as error:
        st.error(f"No fue posible cargar el corpus: {error}")
        st.stop()

    st.sidebar.markdown("### Explorar el corpus")
    st.sidebar.metric("Artículos científicos", len(articulos), help="Corpus documental de la asignatura")
    articulo_sidebar = st.sidebar.selectbox(
        "Vista rápida de un artículo",
        options=sorted(articulos),
        format_func=lambda identificador: f"{identificador:02d} · {articulos[identificador]['titulo']}",
        key="sidebar_article_preview",
    )
    ficha = articulos[articulo_sidebar]
    st.sidebar.markdown(f"**{ficha['titulo']}**")
    st.sidebar.caption(f"Autores: {ficha['autores']}")
    with st.sidebar.expander("Leer abstract original"):
        abstract = ficha["abstract"]
        st.write(abstract if len(abstract) <= 1400 else f"{abstract[:1400].rstrip()}…")
    st.sidebar.divider()
    st.sidebar.caption(
        "El análisis usa el abstract original para buscar patrones y la versión "
        "normalizada para similitud y clustering. El Top 15 quita stopwords."
    )

    columnas = st.columns(4)
    tarjetas = [
        ("01", "Similitud", "Seis medidas para comparar abstracts.", ACCENTOS["lavanda"]),
        ("02", "Clustering", "Tres enlaces y evaluación silhouette.", ACCENTOS["menta"]),
        ("03", "Búsqueda", "KMP, Boyer–Moore y Rabin–Karp.", ACCENTOS["coral"]),
        ("04", "Top 15", "Sorts sobre tuplas término–frecuencia.", ACCENTOS["ambar"]),
    ]
    for columna, (numero, titulo, descripcion, color) in zip(columnas, tarjetas):
        columna.markdown(
            f"""
            <div class="feature-card">
              <div class="feature-number" style="color:{color}">{numero} / EXPLORAR</div>
              <h3>{titulo}</h3>
              <p>{descripcion}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    paginas = st.tabs(["Similitud", "Clustering", "Búsqueda", "Top 15"])
    with paginas[0]:
        _render_similitud(articulos)
    with paginas[1]:
        _render_clustering(articulos)
    with paginas[2]:
        _render_busqueda(articulos)
    with paginas[3]:
        _render_frecuencias(articulos)

    st.divider()
    st.caption(
        "Los algoritmos de similitud, clustering, búsqueda y ordenamiento se "
        "implementan en los módulos del proyecto y se reutilizan desde esta interfaz."
    )


if __name__ == "__main__":
    main()
