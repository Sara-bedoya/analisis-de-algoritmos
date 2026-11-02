"""Radix Sort LSD estable sobre tuplas (frecuencia, término).

Complejidad (n = términos, d = dígitos decimales de la frecuencia máxima):
    - Tiempo: O(d · (n + b)) con base b = 10. Cada pasada es un counting
      sort por un dígito.
    - Espacio: O(n + b) por pasada (salida + cubetas 0..9).
La clave numérica es la frecuencia; el término viaja pegado a ella.
"""

from __future__ import annotations

from src.sorting.tipos import TerminoFrecuencia, validar_terminos


def ordenar_por_digito(
    datos: list[TerminoFrecuencia],
    exponente: int,
) -> list[TerminoFrecuencia]:
    """Counting sort estable según el dígito ``(frecuencia // exponente) % 10``.

    Args:
        datos: Tuplas a reordenar.
        exponente: 1 = unidades, 10 = decenas, 100 = centenas, etc.

    Returns:
        Nueva lista estable respecto al dígito actual.
    """
    cantidad = len(datos)
    resultado: list[TerminoFrecuencia | None] = [None] * cantidad
    conteo = [0] * 10  # base decimal: O(b) = O(1) aquí.

    # O(n): cuenta apariciones del dígito.
    for frecuencia, _palabra in datos:
        digito = (frecuencia // exponente) % 10
        conteo[digito] += 1

    # O(b): posiciones acumuladas.
    for indice in range(1, 10):
        conteo[indice] += conteo[indice - 1]

    # O(n): coloca la tupla completa, de derecha a izquierda (estabilidad).
    for item in reversed(datos):
        digito = (item[0] // exponente) % 10
        resultado[conteo[digito] - 1] = item
        conteo[digito] -= 1

    return [item for item in resultado if item is not None]


def radix_sort(datos: list[TerminoFrecuencia]) -> list[TerminoFrecuencia]:
    """Ordena las tuplas por frecuencia con Radix Sort LSD.

    Args:
        datos: Lista de tuplas ``(frecuencia, término)``.

    Returns:
        Nueva lista ordenada de menor a mayor frecuencia.
    """
    if not datos:
        return []

    validar_terminos(datos)

    resultado = list(datos)
    maximo = max(item[0] for item in resultado)
    exponente = 1

    # d pasadas: mientras queden dígitos en k_max. Tiempo O(d (n + 10)).
    while maximo // exponente > 0:
        resultado = ordenar_por_digito(resultado, exponente)
        exponente *= 10

    return resultado
