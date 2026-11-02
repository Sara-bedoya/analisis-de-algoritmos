"""Lectura de PDF y extracción de abstracts del corpus."""

from __future__ import annotations

import logging
import re
from pathlib import Path

import pymupdf

logger = logging.getLogger(__name__)


def extraer_texto_pdf(ruta_pdf: str | Path) -> str:
    """Extrae el texto completo de un archivo PDF.

    Args:
        ruta_pdf: Ruta del archivo PDF.

    Returns:
        Texto concatenado de todas las páginas.

    Raises:
        FileNotFoundError: Si el archivo no existe.
        RuntimeError: Si PyMuPDF no puede abrir o leer el documento.
    """
    ruta = Path(ruta_pdf)
    if not ruta.is_file():
        raise FileNotFoundError(f"No existe el PDF: {ruta}")

    try:
        with pymupdf.open(ruta) as documento:
            partes_texto = [pagina.get_text() for pagina in documento]
    except FileNotFoundError:
        raise
    except Exception as error:
        logger.exception("No se pudo leer %s", ruta)
        raise RuntimeError(f"No se pudo leer el PDF {ruta.name}: {error}") from error

    return "\n".join(partes_texto)


def extraer_abstract(texto: str | None) -> str | None:
    """Localiza el abstract y recorta hasta Keywords o Introduction.

    Args:
        texto: Texto plano extraído del PDF.

    Returns:
        Abstract recortado, o ``None`` si no hay límites reconocibles.
    """
    if not texto or not str(texto).strip():
        logger.warning("Se pidió extraer abstract de un texto vacío.")
        return None

    # Caso especial del artículo 7:
    # las palabras clave aparecen antes del resumen.
    marcador_7 = "The rapid expansion of Learning Analytics"

    if marcador_7 in texto:
        inicio = texto.find(marcador_7)
        patron_final_7 = re.compile(
            r"(?im)^[ \t]*1\.\s*Introduction[ \t]*$"
        )
        final = patron_final_7.search(texto, inicio)

        if final:
            return texto[inicio:final.start()].strip()

    # Caso especial del artículo 9:
    # el resumen termina antes del primer párrafo
    # de la introducción, que no tiene encabezado visible.
    marcador_9 = "From unique educational perspectives"
    final_9 = "Since Alan Turing first articulated"

    if marcador_9 in texto and final_9 in texto:
        inicio = texto.find(marcador_9)
        fin = texto.find(final_9, inicio)

        if fin > inicio:
            return texto[inicio:fin].strip()

    # Caso especial del artículo 17:
    # el encabezado Keywords incluye los términos
    # en la misma línea.
    marcador_17 = (
        "Artificial intelligence (AI), as an emerging technology"
    )

    if marcador_17 in texto:
        inicio = texto.find(marcador_17)
        patron_final_17 = re.compile(
            r"(?im)^[ \t]*Keywords\b"
        )
        final = patron_final_17.search(texto, inicio)

        if final:
            return texto[inicio:final.start()].strip()


    # Reconocemos los encabezados con y sin espacios.
    patron_inicio = re.compile(
        r"(?im)^[ \t]*(?:"
        r"A\s*B\s*S\s*T\s*R\s*A\s*C\s*T"
        r"|Abstract"
        r")[ \t]*:?[ \t]*$"
    )
    inicio_encontrado = patron_inicio.search(texto)

    # Algunos PDF ponen Abstract y el resumen en la misma línea.
    if inicio_encontrado is None:
        patron_inicio = re.compile(
            r"(?i)\b(?:A\s*B\s*T\s*R\s*A\s*C\s*T|Abstract)"
        )
        inicio_encontrado = patron_inicio.search(texto)

    if inicio_encontrado is None:
        return None

    inicio = inicio_encontrado.end()

    # Buscamos los posibles límites posteriores al encabezado.
    patron_final = re.compile(
        r"(?im)^[ \t]*(?:"
        r"K\s*E\s*Y\s*W\s*O\s*R\s*D\s*S"
        r"|Keywords?"
        r"|Key words"
        r")[^\r\n]*"
        r"|^[ \t]*\d+(?:\.\d+)*\.?[ \t]*"
        r"INTRODUCTION[^\r\n]*$"
    )

    final_encontrado = patron_final.search(texto, inicio)

    if final_encontrado is None:
        return None

    abstract = texto[inicio:final_encontrado.start()].strip()
    # Eliminamos signos sobrantes al comienzo del resumen.
    abstract = abstract.lstrip(" :\t\n\r")
    
    if not abstract:
        return None

    return abstract
