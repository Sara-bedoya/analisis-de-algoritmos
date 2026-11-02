"""Obtiene las 15 palabras más frecuentes desde el arreglo de cada algoritmo."""

from __future__ import annotations

import csv
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.sorting.service import cargar_pares, ranking_por_algoritmo

logger = logging.getLogger(__name__)

RUTA_ENTRADA = Path("data/frecuencias_palabras.csv")
RUTA_SALIDA = Path("data/top_15_algoritmos.csv")


def main() -> None:
    """Carga pares, ordena tuplas completas y guarda el Top 15 de cada sort."""
    pares = cargar_pares(RUTA_ENTRADA)
    resultados = ranking_por_algoritmo(pares, n=15)

    coinciden = (
        resultados["Counting Sort"]
        == resultados["Radix Sort"]
        == resultados["Bucket Sort"]
    )

    print("========== TOP 15 POR ALGORITMO ==========")
    for nombre, top_15 in resultados.items():
        print(f"\n{nombre}")
        print(f"{'Posición':<10}{'Palabra':<20}{'Frecuencia':>12}")
        for posicion, (frecuencia, palabra) in enumerate(top_15, start=1):
            print(f"{posicion:<10}{palabra:<20}{frecuencia:>12}")

    print(f"\nLos tres resultados coinciden: {coinciden}")

    RUTA_SALIDA.parent.mkdir(parents=True, exist_ok=True)
    with RUTA_SALIDA.open("w", encoding="utf-8", newline="") as archivo:
        escritor = csv.writer(archivo)
        escritor.writerow(["algoritmo", "posicion", "palabra", "frecuencia"])
        for nombre, top_15 in resultados.items():
            for posicion, (frecuencia, palabra) in enumerate(top_15, start=1):
                escritor.writerow([nombre, posicion, palabra, frecuencia])

    print(f"Resultados guardados en: {RUTA_SALIDA}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    main()
