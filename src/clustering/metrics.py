"""Métricas internas de evaluación para clustering jerárquico.

Incluye:
- Coeficiente Silhouette promedio para evaluar la cohesión/separación de cortes planos.
- Coeficiente de Correlación Cofenética (CPCC) para evaluar la fidelidad estructural
  del dendrograma completo respecto a la matriz de distancias original (Sokal & Rohlf, 1962).
"""

from __future__ import annotations

from math import sqrt


def silhouette_promedio(
    matriz_distancias: list[list[float]],
    etiquetas: list[int],
) -> float:
    """Calcula el coeficiente silhouette promedio desde la matriz de distancias."""
    cantidad = len(matriz_distancias)
    if len(etiquetas) != cantidad:
        raise ValueError("Debe existir una etiqueta por elemento de la matriz.")

    grupos: dict[int, list[int]] = {}
    for indice, etiqueta in enumerate(etiquetas):
        grupos.setdefault(etiqueta, []).append(indice)
    if len(grupos) < 2 or len(grupos) >= cantidad:
        raise ValueError("Silhouette requiere entre 2 y n-1 grupos.")

    siluetas = []
    for indice, etiqueta in enumerate(etiquetas):
        grupo_propio = grupos[etiqueta]
        if len(grupo_propio) == 1:
            siluetas.append(0.0)
            continue

        distancia_interna = sum(
            matriz_distancias[indice][otro]
            for otro in grupo_propio
            if otro != indice
        ) / (len(grupo_propio) - 1)

        distancia_otro_grupo = min(
            sum(matriz_distancias[indice][otro] for otro in grupo)
            / len(grupo)
            for otra_etiqueta, grupo in grupos.items()
            if otra_etiqueta != etiqueta
        )

        denominador = max(distancia_interna, distancia_otro_grupo)
        siluetas.append(
            0.0
            if denominador == 0
            else (distancia_otro_grupo - distancia_interna) / denominador
        )

    return sum(siluetas) / cantidad


def matriz_cofenetetica(cantidad_hojas: int, fusiones: list) -> list[list[float]]:
    """Calcula la matriz de distancias cofenéticas C(i, j) desde el historial de fusiones.

    La distancia cofenética entre dos hojas i y j es la altura en el dendrograma
    (distancia de enlace) en la que ambos elementos se unen por primera vez en el mismo cluster.

    Args:
        cantidad_hojas: Número n de elementos base (hojas 0..n-1).
        fusiones: Historial de n-1 fusiones de tipo Merge(left, right, distance, size).

    Returns:
        Matriz simétrica de tamaño n x n con las distancias cofenéticas.
    """
    clusters: dict[int, list[int]] = {indice: [indice] for indice in range(cantidad_hojas)}
    matriz_c = [[0.0 for _ in range(cantidad_hojas)] for _ in range(cantidad_hojas)]

    for indice_fusion, fusion in enumerate(fusiones):
        hojas_izq = clusters.pop(fusion.left)
        hojas_der = clusters.pop(fusion.right)
        altura = float(fusion.distance)

        for u in hojas_izq:
            for v in hojas_der:
                matriz_c[u][v] = altura
                matriz_c[v][u] = altura

        clusters[cantidad_hojas + indice_fusion] = hojas_izq + hojas_der

    return matriz_c


def correlacion_cofenetetica(matriz_distancias: list[list[float]], fusiones: list) -> float:
    """Calcula el Coeficiente de Correlación Cofenética (CPCC).

    Mide la correlación de Pearson entre las distancias originales por pares (D)
    y las distancias cofenéticas del dendrograma (C). Un valor cercano a 1 indica
    que el árbol jerárquico preserva con alta fidelidad las distancias reales.

    Fórmula:
        r = sum((D_ij - mean(D)) * (C_ij - mean(C))) / sqrt(sum((D_ij - mean(D))^2) * sum((C_ij - mean(C))^2))

    Args:
        matriz_distancias: Matriz simétrica de distancias originales.
        fusiones: Lista de fusiones generadas por clustering_jerarquico.

    Returns:
        Coeficiente de Pearson r en el intervalo [-1, 1].
    """
    cantidad = len(matriz_distancias)
    if cantidad < 2:
        return 1.0

    matriz_c = matriz_cofenetetica(cantidad, fusiones)

    d_valores = []
    c_valores = []
    for i in range(cantidad):
        for j in range(i + 1, cantidad):
            d_valores.append(float(matriz_distancias[i][j]))
            c_valores.append(float(matriz_c[i][j]))

    total_pares = len(d_valores)
    if total_pares == 0:
        return 1.0

    media_d = sum(d_valores) / total_pares
    media_c = sum(c_valores) / total_pares

    covarianza = sum((d - media_d) * (c - media_c) for d, c in zip(d_valores, c_valores))
    var_d = sum((d - media_d) ** 2 for d in d_valores)
    var_c = sum((c - media_c) ** 2 for c in c_valores)

    denominador = sqrt(var_d * var_c)
    if denominador == 0:
        return 0.0

    return covarianza / denominador
