"""Tipos y validaciones compartidas por los algoritmos de ordenamiento."""

from __future__ import annotations

from typing import Tuple

# (frecuencia, término): la clave numérica va primero; la tupla viaja completa.
TerminoFrecuencia = Tuple[int, str]


def validar_terminos(datos: list[TerminoFrecuencia]) -> None:
    """Comprueba que cada elemento sea ``(frecuencia no negativa, palabra)``.

    Args:
        datos: Colección a ordenar.

    Raises:
        TypeError: Si un elemento no es una tupla ``(int, str)``.
        ValueError: Si alguna frecuencia es negativa.
    """
    for indice, item in enumerate(datos):
        if not isinstance(item, tuple) or len(item) != 2:
            raise TypeError(
                f"El elemento {indice} debe ser una tupla (frecuencia, palabra)."
            )
        frecuencia, palabra = item
        if not isinstance(frecuencia, int) or isinstance(frecuencia, bool):
            raise TypeError(
                f"La frecuencia del elemento {indice} debe ser un entero."
            )
        if not isinstance(palabra, str):
            raise TypeError(
                f"El término del elemento {indice} debe ser una cadena."
            )
        if frecuencia < 0:
            raise ValueError(
                "Los algoritmos requieren frecuencias enteras no negativas."
            )


def extraer_top(ordenados: list[TerminoFrecuencia], n: int = 15) -> list[TerminoFrecuencia]:
    """Obtiene los n términos de mayor frecuencia desde un arreglo ya ordenado.

    Los algoritmos dejan el arreglo en orden ascendente. El ranking se recorta
    del final; no se consulta ningún diccionario externo.

    Args:
        ordenados: Tuplas ordenadas de menor a mayor frecuencia.
        n: Tamaño del ranking.

    Returns:
        Hasta ``n`` tuplas de mayor a menor frecuencia.
    """
    if n <= 0 or not ordenados:
        return []
    return list(reversed(ordenados[-n:]))
