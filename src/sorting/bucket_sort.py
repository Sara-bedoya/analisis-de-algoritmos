"""Bucket Sort sobre tuplas (frecuencia, término).

Complejidad (n = términos, b = n cubetas):
    - Distribución: O(n).
    - Ordenamiento interno: insertion sort por cubeta.
      Promedio O(n) si las frecuencias se reparte de forma uniforme
      (n elementos en n cubetas ≈ O(1) por cubeta).
      Peor caso O(n²) si casi todos caen en la misma cubeta.
    - Espacio: O(n).
La comparación del insertion sort usa solo ``item[0]`` (frecuencia);
el intercambio mueve la tupla completa.
"""

from __future__ import annotations

from src.sorting.tipos import TerminoFrecuencia, validar_terminos


def _insercion(bucket: list[TerminoFrecuencia]) -> None:
    """Insertion sort in-place por frecuencia. O(m²) en una cubeta de m elementos."""
    for i in range(1, len(bucket)):
        actual = bucket[i]
        j = i - 1
        while j >= 0 and bucket[j][0] > actual[0]:
            bucket[j + 1] = bucket[j]
            j -= 1
        bucket[j + 1] = actual


def bucket_sort(datos: list[TerminoFrecuencia]) -> list[TerminoFrecuencia]:
    """Distribuye tuplas en cubetas y las fusiona en orden ascendente.

    Args:
        datos: Lista de tuplas ``(frecuencia, término)``.

    Returns:
        Nueva lista ordenada. Lista vacía si no hay datos.
    """
    if not datos:
        return []

    validar_terminos(datos)

    minimo = min(item[0] for item in datos)
    maximo = max(item[0] for item in datos)
    cantidad_buckets = len(datos)
    buckets: list[list[TerminoFrecuencia]] = [[] for _ in range(cantidad_buckets)]
    rango = maximo - minimo + 1

    # O(n): cada tupla completa entra en una cubeta indexada por frecuencia.
    for item in datos:
        indice = (item[0] - minimo) * cantidad_buckets // rango
        indice = min(indice, cantidad_buckets - 1)
        buckets[indice].append(item)

    resultado: list[TerminoFrecuencia] = []
    for bucket in buckets:
        _insercion(bucket)
        resultado.extend(bucket)

    return resultado
