"""Carga pares término-frecuencia y obtiene el Top 15 desde cada algoritmo."""

from __future__ import annotations

import csv
import logging
from collections.abc import Callable
from pathlib import Path

from src.preprocessing.frecuencia_palabras import frecuencias_desde_textos
from src.sorting.bucket_sort import bucket_sort
from src.sorting.counting_sort import counting_sort
from src.sorting.radix_sort import radix_sort
from src.sorting.tipos import TerminoFrecuencia, extraer_top

logger = logging.getLogger(__name__)

ALGORITMOS: dict[str, Callable[[list[TerminoFrecuencia]], list[TerminoFrecuencia]]] = {
    "Counting Sort": counting_sort,
    "Radix Sort": radix_sort,
    "Bucket Sort": bucket_sort,
}


def pares_desde_contador(frecuencias: dict[str, int]) -> list[TerminoFrecuencia]:
    """Convierte un diccionario palabra → frecuencia en tuplas ordenables.

    Args:
        frecuencias: Conteos de términos.

    Returns:
        Lista ``[(frecuencia, palabra), ...]`` en el orden de inserción.
    """
    return [(cantidad, palabra) for palabra, cantidad in frecuencias.items()]


def cargar_pares(ruta: Path) -> list[TerminoFrecuencia]:
    """Lee ``palabra,frecuencia`` desde CSV.

    Args:
        ruta: Archivo generado por el módulo de frecuencias.

    Returns:
        Lista de tuplas listas para los algoritmos.

    Raises:
        FileNotFoundError: Si el CSV no existe.
        ValueError: Si está vacío o mal formado.
    """
    if not ruta.is_file():
        raise FileNotFoundError(
            f"No existe {ruta}. Genera las frecuencias antes de ordenar."
        )

    pares: list[TerminoFrecuencia] = []
    with ruta.open("r", encoding="utf-8", newline="") as archivo:
        lector = csv.DictReader(archivo)
        requeridos = {"palabra", "frecuencia"}
        faltantes = requeridos - set(lector.fieldnames or [])
        if faltantes:
            raise ValueError(f"Al CSV le faltan columnas: {sorted(faltantes)}")

        for numero, fila in enumerate(lector, start=2):
            palabra = (fila.get("palabra") or "").strip()
            crudo = (fila.get("frecuencia") or "").strip()
            if not palabra or not crudo:
                logger.warning("Fila %s incompleta; se omite.", numero)
                continue
            try:
                pares.append((int(crudo), palabra))
            except ValueError as error:
                raise ValueError(
                    f"Frecuencia no entera en la fila {numero}: {crudo!r}."
                ) from error

    if not pares:
        raise ValueError(f"El archivo {ruta} no contiene términos para ordenar.")
    return pares


def pares_desde_articulos(articulos: dict) -> list[TerminoFrecuencia]:
    """Cuenta términos de contenido en los abstracts del corpus.

    Usa el abstract original si existe; si no, el normalizado.

    Args:
        articulos: Catálogo indexado por id.

    Returns:
        Tuplas ``(frecuencia, término)``.
    """
    textos = []
    for articulo in articulos.values():
        texto = articulo.get("abstract") or articulo.get("abstract_normalizado") or ""
        textos.append(texto)
    return pares_desde_contador(frecuencias_desde_textos(textos))


def ranking_por_algoritmo(
    pares: list[TerminoFrecuencia],
    n: int = 15,
) -> dict[str, list[TerminoFrecuencia]]:
    """Ejecuta los tres algoritmos y recorta el Top n del arreglo ordenado.

    Args:
        pares: Tuplas a ordenar.
        n: Tamaño del ranking.

    Returns:
        Nombre de algoritmo → ranking de mayor a menor frecuencia.

    Raises:
        ValueError: Si no hay pares.
    """
    if not pares:
        raise ValueError("No hay términos para ordenar.")

    resultados: dict[str, list[TerminoFrecuencia]] = {}
    for nombre, algoritmo in ALGORITMOS.items():
        ordenados = algoritmo(pares)
        resultados[nombre] = extraer_top(ordenados, n)
    return resultados
