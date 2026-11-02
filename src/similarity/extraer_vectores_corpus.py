"""Script utilitario para extraer vectores del vocabulario del corpus.

Permite generar archivos compactos (~100 KB para GloVe, ~500 KB para FastText)
que pueden subirse a GitHub y ejecutarse en Streamlit Community Cloud sin
necesidad de cargar los archivos originales de más de 1 GB.
"""

from __future__ import annotations

import gzip
import logging
from pathlib import Path

from src.similarity.corpus import cargar_corpus

logger = logging.getLogger(__name__)

RUTA_RAIZ = Path(__file__).resolve().parents[2]
RUTA_MODELS = RUTA_RAIZ / "data" / "models"
CORPUS_CSV = RUTA_RAIZ / "data" / "corpus_articulos.csv"


def obtener_vocabulario_corpus() -> set[str]:
    """Extrae el conjunto de palabras únicas presentes en los abstracts normalizados."""
    articulos = cargar_corpus(CORPUS_CSV)
    vocabulario = set()
    for art in articulos.values():
        vocabulario.update(art["abstract_normalizado"].split())
    return vocabulario


def extraer_subconjunto(
    ruta_origen: Path,
    ruta_destino: Path,
    vocabulario: set[str],
) -> int:
    """Extrae únicamente los vectores de palabras que pertenecen al vocabulario.

    Args:
        ruta_origen: Archivo descargado original (.gz o texto plano).
        ruta_destino: Archivo .gz compacto resultante.
        vocabulario: Conjunto de palabras a conservar.

    Returns:
        Cantidad de vectores guardados.
    """
    if not ruta_origen.is_file():
        logger.warning("No se encontró el archivo origen: %s", ruta_origen)
        return 0

    ruta_destino.parent.mkdir(parents=True, exist_ok=True)
    abrir_in = gzip.open if ruta_origen.suffix.lower() == ".gz" else open
    lineas_guardadas = []
    dimension = None

    print(f"Procesando {ruta_origen.name}...", flush=True)
    with abrir_in(ruta_origen, "rt", encoding="utf-8", errors="replace") as archivo:
        primera_linea = archivo.readline()
        campos = primera_linea.split()
        if len(campos) == 2 and campos[0].isdigit() and campos[1].isdigit():
            dimension = int(campos[1])
        else:
            token = campos[0]
            if token in vocabulario:
                lineas_guardadas.append(primera_linea.strip())
            dimension = len(campos) - 1

        for numero, linea in enumerate(archivo, start=2):
            campos = linea.split(maxsplit=1)
            if campos and campos[0] in vocabulario:
                lineas_guardadas.append(linea.strip())

    with gzip.open(ruta_destino, "wt", encoding="utf-8") as salida:
        salida.write(f"{len(lineas_guardadas)} {dimension}\n")
        for linea in lineas_guardadas:
            salida.write(linea + "\n")

    print(
        f"✓ Guardados {len(lineas_guardadas)}/{len(vocabulario)} vectores en {ruta_destino.name} "
        f"(Dimensión {dimension})"
    )
    return len(lineas_guardadas)


def main() -> None:
    vocabulario = obtener_vocabulario_corpus()
    print(f"Vocabulario del corpus: {len(vocabulario)} términos únicos.")

    # 1. GloVe
    origen_glove = RUTA_MODELS / "glove-wiki-gigaword-50.gz"
    destino_glove = RUTA_MODELS / "glove_corpus.txt.gz"
    extraer_subconjunto(origen_glove, destino_glove, vocabulario)

    # 2. FastText
    origen_fasttext = RUTA_MODELS / "fasttext-wiki-news-subwords-300.gz"
    destino_fasttext = RUTA_MODELS / "fasttext_corpus.txt.gz"
    extraer_subconjunto(origen_fasttext, destino_fasttext, vocabulario)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    main()
