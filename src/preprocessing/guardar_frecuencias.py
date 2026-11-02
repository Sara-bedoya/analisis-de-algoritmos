"""Guarda las frecuencias de términos de contenido en CSV."""

from __future__ import annotations

import csv
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.preprocessing.frecuencia_palabras import (
    calcular_frecuencias,
    frecuencias_desde_textos,
)
from src.similarity.corpus import cargar_corpus

logger = logging.getLogger(__name__)


def guardar_frecuencias() -> Path:
    """Calcula frecuencias y escribe ``data/frecuencias_palabras.csv``.

    Prefiere los PDF de ``data/articulos``. Si no hay PDF, usa los abstracts
    del catálogo ``data/corpus_articulos.csv``.

    Returns:
        Ruta del CSV generado.
    """
    carpeta_pdf = Path("data/articulos")
    if carpeta_pdf.is_dir() and any(carpeta_pdf.glob("*.pdf")):
        frecuencias, procesados, total = calcular_frecuencias(carpeta_pdf)
    else:
        logger.warning("No hay PDF; se cuentan términos desde el catálogo CSV.")
        articulos = cargar_corpus()
        textos = [articulo["abstract"] for articulo in articulos.values()]
        frecuencias = frecuencias_desde_textos(textos)
        procesados = total = len(articulos)
    carpeta_salida = Path("data")
    carpeta_salida.mkdir(exist_ok=True)
    ruta_salida = carpeta_salida / "frecuencias_palabras.csv"

    with ruta_salida.open("w", newline="", encoding="utf-8") as archivo:
        escritor = csv.writer(archivo)
        escritor.writerow(["palabra", "frecuencia"])
        for palabra, cantidad in frecuencias.items():
            escritor.writerow([palabra, cantidad])

    logger.info("Artículos procesados: %s/%s", procesados, total)
    print(f"Artículos procesados: {procesados}/{total}")
    print(f"Palabras diferentes: {len(frecuencias)}")
    print(f"Archivo guardado en: {ruta_salida}")
    return ruta_salida


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    guardar_frecuencias()
