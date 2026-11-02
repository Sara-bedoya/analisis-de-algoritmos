"""Interfaz de consola para generar clustering, métricas y dendrogramas."""

import argparse
import csv
from pathlib import Path

from src.clustering.hierarchical import METODOS, cortar_arbol
from src.clustering.service import evaluar_enlaces, matriz_distancias_tfidf
from src.clustering.visualization import guardar_dendrograma
from src.similarity.corpus import cargar_corpus

RAIZ_PROYECTO = Path(__file__).resolve().parents[2]
RUTA_SALIDA_PREDETERMINADA = RAIZ_PROYECTO / "output" / "clustering"


def crear_parser():
    parser = argparse.ArgumentParser(
        description="Agrupa los abstracts con tres enlaces jerárquicos desde cero."
    )
    parser.add_argument(
        "--salida",
        type=Path,
        default=RUTA_SALIDA_PREDETERMINADA,
        help="Carpeta donde guardar dendrogramas y CSV de resultados.",
    )
    return parser


def main(argv=None):
    parser = crear_parser()
    argumentos = parser.parse_args(argv)
    try:
        articulos = cargar_corpus()
        ids, matriz_distancias = matriz_distancias_tfidf(articulos)
        resultados = evaluar_enlaces(matriz_distancias)
        _guardar_matriz(argumentos.salida / "distancias_coseno.csv", ids, matriz_distancias)
        _guardar_puntuaciones(argumentos.salida / "silhouette.csv", resultados)
        _guardar_resumen_metricas(argumentos.salida / "metricas_coherencia.csv", resultados)
        _guardar_agrupaciones(
            argumentos.salida / "agrupaciones_mejores.csv",
            articulos,
            ids,
            resultados,
        )

        for metodo in METODOS:
            detalle = resultados[metodo]
            guardar_dendrograma(
                articulos,
                ids,
                detalle["fusiones"],
                metodo,
                argumentos.salida / f"dendrograma_{metodo}.png",
            )
            _guardar_fusiones(
                argumentos.salida / f"fusiones_{metodo}.csv",
                detalle["fusiones"],
            )
    except (FileNotFoundError, OSError, ValueError) as error:
        parser.error(str(error))

    print(f"Abstracts agrupados: {len(ids)}")
    print("\n--- Evaluación de Coherencia Cofenética (CPCC) y Silhouette ---")
    for metodo in METODOS:
        detalle = resultados[metodo]
        print(
            f"  {metodo.title():<10} Linkage | CPCC: {detalle['cpcc']:.6f} | "
            f"Mejor k: {detalle['mejor_k']} (Silhouette: {detalle['mejor_silhouette']:.6f})"
        )

    mejor_cpcc = max(resultados[m]["cpcc"] for m in METODOS)
    ganador_cpcc = [m for m in METODOS if resultados[m]["cpcc"] == mejor_cpcc][0]
    print(
        f"\n🏆 Algoritmo con mayor coherencia estructural (CPCC): {ganador_cpcc.title()} Linkage "
        f"({mejor_cpcc:.6f}) - Preserva con mayor fidelidad las distancias reales del corpus."
    )
    print(f"Archivos guardados en: {argumentos.salida}")


def _guardar_matriz(ruta, ids, matriz):
    ruta.parent.mkdir(parents=True, exist_ok=True)
    with ruta.open("w", encoding="utf-8-sig", newline="") as archivo:
        escritor = csv.writer(archivo)
        escritor.writerow(["id", *ids])
        for identificador, fila in zip(ids, matriz):
            escritor.writerow([identificador, *[f"{valor:.12f}" for valor in fila]])


def _guardar_puntuaciones(ruta, resultados):
    ruta.parent.mkdir(parents=True, exist_ok=True)
    with ruta.open("w", encoding="utf-8-sig", newline="") as archivo:
        campos = ["metodo", "k", "silhouette", "seleccionado"]
        escritor = csv.DictWriter(archivo, fieldnames=campos)
        escritor.writeheader()
        for metodo, detalle in resultados.items():
            for cantidad_grupos, puntuacion in detalle["puntuaciones"]:
                escritor.writerow({
                    "metodo": metodo,
                    "k": cantidad_grupos,
                    "silhouette": f"{puntuacion:.12f}",
                    "seleccionado": cantidad_grupos == detalle["mejor_k"],
                })


def _guardar_resumen_metricas(ruta, resultados):
    ruta.parent.mkdir(parents=True, exist_ok=True)
    with ruta.open("w", encoding="utf-8-sig", newline="") as archivo:
        campos = ["metodo", "mejor_k_silhouette", "mejor_silhouette", "cpcc", "interpretacion_cpcc"]
        escritor = csv.DictWriter(archivo, fieldnames=campos)
        escritor.writeheader()
        for metodo, detalle in resultados.items():
            cpcc = detalle["cpcc"]
            interpretacion = (
                "Excelente fidelidad" if cpcc >= 0.8 else
                "Buena fidelidad" if cpcc >= 0.7 else
                "Distorsión moderada"
            )
            escritor.writerow({
                "metodo": metodo,
                "mejor_k_silhouette": detalle["mejor_k"],
                "mejor_silhouette": f"{detalle['mejor_silhouette']:.6f}",
                "cpcc": f"{cpcc:.6f}",
                "interpretacion_cpcc": interpretacion,
            })


def _guardar_fusiones(ruta, fusiones):
    ruta.parent.mkdir(parents=True, exist_ok=True)
    with ruta.open("w", encoding="utf-8-sig", newline="") as archivo:
        campos = ["paso", "left", "right", "distance", "size"]
        escritor = csv.DictWriter(archivo, fieldnames=campos)
        escritor.writeheader()
        for paso, fusion in enumerate(fusiones, start=1):
            escritor.writerow({
                "paso": paso,
                "left": fusion.left,
                "right": fusion.right,
                "distance": f"{fusion.distance:.12f}",
                "size": fusion.size,
            })


def _guardar_agrupaciones(ruta, articulos, ids, resultados):
    ruta.parent.mkdir(parents=True, exist_ok=True)
    with ruta.open("w", encoding="utf-8-sig", newline="") as archivo:
        campos = ["id", "titulo", *[f"grupo_{metodo}" for metodo in METODOS]]
        escritor = csv.DictWriter(archivo, fieldnames=campos)
        escritor.writeheader()
        asignaciones = {}
        for metodo in METODOS:
            detalle = resultados[metodo]
            etiquetas = cortar_arbol(len(ids), detalle["fusiones"], detalle["mejor_k"])
            asignaciones[metodo] = {identificador: etiqueta + 1 for identificador, etiqueta in zip(ids, etiquetas)}

        for identificador in ids:
            fila = {
                "id": identificador,
                "titulo": articulos[identificador]["titulo"],
            }
            for metodo in METODOS:
                fila[f"grupo_{metodo}"] = asignaciones[metodo][identificador]
            escritor.writerow(fila)


if __name__ == "__main__":
    main()
