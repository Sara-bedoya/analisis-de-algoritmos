"""Compara Counting, Radix y Bucket Sort sobre las mismas tuplas."""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from statistics import median

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.sorting.service import ALGORITMOS, cargar_pares
from src.sorting.tipos import TerminoFrecuencia

logger = logging.getLogger(__name__)

RUTA_CSV = Path("data/frecuencias_palabras.csv")
REPETICIONES = 7
EJECUCIONES_POR_MEDICION = 100


def _esta_ordenado(datos: list[TerminoFrecuencia]) -> bool:
    """Verifica orden no decreciente por frecuencia."""
    return all(datos[i][0] <= datos[i + 1][0] for i in range(len(datos) - 1))


def medir_tiempo(algoritmo, datos: list[TerminoFrecuencia]) -> float:
    """Mediana del tiempo por ejecución en milisegundos.

    Args:
        algoritmo: Función de ordenamiento.
        datos: Tuplas de entrada (no se mutan).

    Returns:
        Mediana en milisegundos.
    """
    algoritmo(datos)
    mediciones: list[float] = []
    for _ in range(REPETICIONES):
        inicio = time.perf_counter()
        for _ in range(EJECUCIONES_POR_MEDICION):
            algoritmo(datos)
        fin = time.perf_counter()
        mediciones.append((fin - inicio) * 1000 / EJECUCIONES_POR_MEDICION)
    return median(mediciones)


def main() -> None:
    """Mide tiempo y comprueba que los tres sorts produzcan el mismo arreglo."""
    pares = cargar_pares(RUTA_CSV)
    resultados: dict[str, list[TerminoFrecuencia]] = {}

    print("========== COMPARACIÓN DE ALGORITMOS ==========")
    print(f"Términos procesados: {len(pares)}")
    print(f"Repeticiones por algoritmo: {REPETICIONES}")
    print(f"Ejecuciones por medición: {EJECUCIONES_POR_MEDICION}\n")

    for nombre, algoritmo in ALGORITMOS.items():
        ordenados = algoritmo(pares)
        correcto = len(ordenados) == len(pares) and _esta_ordenado(ordenados)
        resultados[nombre] = ordenados
        tiempo = medir_tiempo(algoritmo, pares)
        print(nombre)
        print(f"  Ordenamiento correcto: {correcto}")
        print(f"  Mediana por ejecución: {tiempo:.6f} ms\n")

    todos_iguales = (
        resultados["Counting Sort"]
        == resultados["Radix Sort"]
        == resultados["Bucket Sort"]
    )
    print(f"Los tres resultados coinciden: {todos_iguales}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    main()
