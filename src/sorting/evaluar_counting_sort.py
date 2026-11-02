"""Evalúa Counting Sort con las frecuencias reales del corpus."""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.sorting.counting_sort import counting_sort
from src.sorting.service import cargar_pares
from src.sorting.tipos import extraer_top

RUTA_CSV = Path("data/frecuencias_palabras.csv")


def evaluar_counting_sort() -> None:
    """Ordena tuplas, valida el orden y muestra el Top 15 del arreglo."""
    pares = cargar_pares(RUTA_CSV)

    inicio = time.perf_counter()
    resultado = counting_sort(pares)
    tiempo_ms = (time.perf_counter() - inicio) * 1000

    correcto = all(
        resultado[i][0] <= resultado[i + 1][0] for i in range(len(resultado) - 1)
    )

    print("========== EVALUACIÓN COUNTING SORT ==========")
    print(f"Palabras procesadas: {len(pares)}")
    print(f"Ordenamiento correcto: {correcto}")
    print(f"Tiempo de ejecución: {tiempo_ms:.6f} ms")
    print("\n15 palabras más frecuentes:")
    for frecuencia, palabra in extraer_top(resultado, 15):
        print(f"{palabra}: {frecuencia}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    evaluar_counting_sort()
