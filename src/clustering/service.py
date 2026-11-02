"""Prepara distancias, construye árboles y evalúa sus cortes y coherencia."""

from src.clustering.hierarchical import METODOS, clustering_jerarquico, cortar_arbol
from src.clustering.metrics import correlacion_cofenetetica, silhouette_promedio
from src.similarity.classic import construir_tfidf, similitud_coseno_vectores


def matriz_distancias_tfidf(articulos):
    """Crea distancia coseno 1-coseno entre los TF-IDF de todos los abstracts."""
    if not articulos:
        raise ValueError("No hay artículos para construir la matriz de distancias.")
    ids = sorted(articulos)
    documentos = [articulos[identificador]["abstract_normalizado"] for identificador in ids]
    if any(not (documento or "").strip() for documento in documentos):
        raise ValueError("Hay abstracts normalizados vacíos; no se puede calcular TF-IDF.")
    vectores = construir_tfidf(documentos)
    matriz = [[0.0 for _ in ids] for _ in ids]
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            coseno = similitud_coseno_vectores(vectores[i], vectores[j])
            coseno = min(1.0, max(0.0, coseno))
            distancia = 1.0 - coseno
            matriz[i][j] = distancia
            matriz[j][i] = distancia
    return ids, matriz


def evaluar_enlaces(matriz_distancias):
    """Evalúa silhouette y correlación cofenética (CPCC) en los tres árboles jerárquicos."""
    cantidad = len(matriz_distancias)
    if cantidad < 3:
        raise ValueError("Se requieren al menos tres abstracts para evaluar clustering.")

    resultados = {}
    for metodo in METODOS:
        fusiones = clustering_jerarquico(matriz_distancias, metodo)
        puntuaciones = []
        for cantidad_grupos in range(2, cantidad):
            etiquetas = cortar_arbol(cantidad, fusiones, cantidad_grupos)
            puntuacion = silhouette_promedio(matriz_distancias, etiquetas)
            puntuaciones.append((cantidad_grupos, puntuacion))
        mejor_k, mejor_silhouette = max(
            puntuaciones,
            key=lambda resultado: (resultado[1], -resultado[0]),
        )
        cpcc = correlacion_cofenetetica(matriz_distancias, fusiones)
        resultados[metodo] = {
            "fusiones": fusiones,
            "puntuaciones": puntuaciones,
            "mejor_k": mejor_k,
            "mejor_silhouette": mejor_silhouette,
            "cpcc": cpcc,
        }
    return resultados
