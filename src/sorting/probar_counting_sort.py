"""Prueba Counting Sort con tuplas (frecuencia, término), incluida estabilidad."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.sorting.counting_sort import counting_sort

datos = [(4, "alfa"), (2, "beta"), (4, "gamma"), (1, "delta"), (2, "epsilon"), (3, "zeta")]
esperado = [(1, "delta"), (2, "beta"), (2, "epsilon"), (3, "zeta"), (4, "alfa"), (4, "gamma")]

resultado = counting_sort(datos)
print("Datos originales:", datos)
print("Datos ordenados:", resultado)
print("Lista vacía:", counting_sort([]))

if resultado == esperado:
    print("Prueba superada correctamente.")
else:
    print("Error en el algoritmo.")
