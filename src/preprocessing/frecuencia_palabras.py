"""Cuenta frecuencias de términos de contenido en los abstracts."""

from __future__ import annotations

import logging
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.extraction.pdf_reader import extraer_abstract, extraer_texto_pdf
from src.preprocessing.limpieza_texto import tokenizar_para_frecuencias

logger = logging.getLogger(__name__)

CARPETA_ARTICULOS = Path("data/articulos")


def calcular_frecuencias(
    carpeta_articulos: Path | None = None,
) -> tuple[Counter[str], int, int]:
    """Extrae abstracts, tokeniza y cuenta términos sin stopwords.

    Args:
        carpeta_articulos: Directorio de PDF. Por defecto ``data/articulos``.

    Returns:
        Tupla ``(frecuencias, articulos_procesados, total_pdf)``.

    Raises:
        FileNotFoundError: Si la carpeta de artículos no existe.
        RuntimeError: Si ningún abstract pudo procesarse.
    """
    carpeta = carpeta_articulos or CARPETA_ARTICULOS
    if not carpeta.is_dir():
        raise FileNotFoundError(
            f"No existe la carpeta de artículos: {carpeta}. "
            "Coloca los PDF del corpus antes de calcular frecuencias."
        )

    frecuencias: Counter[str] = Counter()
    articulos_procesados = 0
    archivos_pdf = sorted(carpeta.glob("*.pdf"))

    if not archivos_pdf:
        raise FileNotFoundError(f"No se encontraron PDF en {carpeta}.")

    for archivo in archivos_pdf:
        try:
            texto_pdf = extraer_texto_pdf(str(archivo))
            abstract = extraer_abstract(texto_pdf)
        except (OSError, RuntimeError, ValueError) as error:
            logger.error("Fallo al leer %s: %s", archivo.name, error)
            continue

        if abstract is None:
            logger.warning("No se pudo extraer el abstract: %s", archivo.name)
            continue

        palabras = tokenizar_para_frecuencias(abstract)
        if not palabras:
            logger.warning("El abstract de %s quedó vacío tras limpiar.", archivo.name)
            continue

        frecuencias.update(palabras)
        articulos_procesados += 1

    if articulos_procesados == 0:
        raise RuntimeError(
            "Ningún abstract pudo procesarse. Revisa la extracción de PDF."
        )

    return frecuencias, articulos_procesados, len(archivos_pdf)


def frecuencias_desde_textos(textos: list[str]) -> Counter[str]:
    """Cuenta términos de contenido a partir de abstracts ya extraídos.

    Args:
        textos: Abstracts originales o normalizados.

    Returns:
        Contador palabra → frecuencia.

    Raises:
        ValueError: Si no hay textos o todos quedan vacíos tras limpiar.
    """
    if not textos:
        raise ValueError("No hay abstracts para contar frecuencias.")

    frecuencias: Counter[str] = Counter()
    for texto in textos:
        frecuencias.update(tokenizar_para_frecuencias(texto))

    if not frecuencias:
        raise ValueError(
            "Tras quitar stopwords y tokens no alfabéticos no quedó ningún término."
        )
    return frecuencias


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    frecuencias, procesados, total = calcular_frecuencias()

    print(f"PDF encontrados: {total}")
    print(f"Artículos procesados: {procesados}")
    print(f"Palabras diferentes: {len(frecuencias)}")
    print("\n15 palabras más frecuentes:")
    for palabra, cantidad in frecuencias.most_common(15):
        print(f"{palabra}: {cantidad}")
