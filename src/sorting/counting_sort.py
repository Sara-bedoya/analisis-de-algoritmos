"""Counting Sort estable sobre tuplas (frecuencia, término).

Complejidad (n = cantidad de términos, k = frecuencia máxima):
    - Tiempo: O(n + k). Un recorrido cuenta claves, otro escribe el
      arreglo de salida y el de prefijos recorre k + 1 cubetas.
    - Espacio: O(n + k). Arreglo de conteo de tamaño k + 1 y copia
      de n elementos.
No compara términos entre sí: solo indexa por la frecuencia entera.
"""

from __future__ import annotations

from src.sorting.tipos import TerminoFrecuencia, validar_terminos


def counting_sort(datos: list[TerminoFrecuencia]) -> list[TerminoFrecuencia]:
    """Ordena de forma estable las tuplas por frecuencia ascendente.

    Cada cubeta mueve el par completo ``(frecuencia, palabra)``. La
    escritura de derecha a izquierda conserva el orden relativo de
    términos con la misma frecuencia.

    Args:
        datos: Lista de tuplas ``(frecuencia, término)``.

    Returns:
        Nueva lista ordenada. Lista vacía si no hay datos.
    """
    if not datos:
        return []

    validar_terminos(datos)

    # k = max(frecuencia). Tiempo O(n) para hallar k.
    maximo = max(item[0] for item in datos)

    # Conteo: O(n + k). conteo[f] = cuántos términos tienen frecuencia f.
    conteo = [0] * (maximo + 1)
    for frecuencia, _palabra in datos:
        conteo[frecuencia] += 1

    # Prefijos: O(k). conteo[f] pasa a ser la posición final exclusiva.
    for indice in range(1, len(conteo)):
        conteo[indice] += conteo[indice - 1]

    # Colocación estable: O(n). Se recorre de derecha a izquierda.
    resultado: list[TerminoFrecuencia | None] = [None] * len(datos)
    for item in reversed(datos):
        frecuencia = item[0]
        resultado[conteo[frecuencia] - 1] = item
        conteo[frecuencia] -= 1

    return [item for item in resultado if item is not None]
