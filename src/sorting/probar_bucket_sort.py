"""Prueba Bucket Sort con tuplas (frecuencia, término)."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.sorting.bucket_sort import bucket_sort

datos = [
    (29, "a"),
    (3, "b"),
    (15, "c"),
    (8, "d"),
    (3, "e"),
    (42, "f"),
    (17, "g"),
    (1, "h"),
]
esperado = [
    (1, "h"),
    (3, "b"),
    (3, "e"),
    (8, "d"),
    (15, "c"),
    (17, "g"),
    (29, "a"),
    (42, "f"),
]

resultado = bucket_sort(datos)
print("Datos originales:", datos)
print("Datos ordenados:", resultado)
print("Lista vacía:", bucket_sort([]))

if resultado == esperado:
    print("Prueba superada correctamente.")
else:
    print("Error en el algoritmo.")
