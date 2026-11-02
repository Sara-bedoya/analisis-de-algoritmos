"""Prueba Radix Sort con tuplas (frecuencia, término)."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.sorting.radix_sort import radix_sort

datos = [
    (170, "a"),
    (45, "b"),
    (75, "c"),
    (90, "d"),
    (802, "e"),
    (24, "f"),
    (2, "g"),
    (66, "h"),
]
esperado = [
    (2, "g"),
    (24, "f"),
    (45, "b"),
    (66, "h"),
    (75, "c"),
    (90, "d"),
    (170, "a"),
    (802, "e"),
]

resultado = radix_sort(datos)
print("Datos originales:", datos)
print("Datos ordenados:", resultado)
print("Lista vacía:", radix_sort([]))

if resultado == esperado:
    print("Prueba superada correctamente.")
else:
    print("Error en el algoritmo.")
