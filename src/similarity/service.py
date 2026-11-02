"""Orquesta los seis enfoques de similitud para todas las parejas seleccionadas."""

from itertools import combinations

from src.similarity.classic import comparar_clasicos, construir_tfidf
from src.similarity.embeddings import cobertura_vocabulario, similitud_embeddings


def comparar_articulos(articulos, identificadores, glove=None, fasttext=None):
    """Compara todos los pares de IDs usando métodos disponibles y cruce de metadatos."""
    identificadores = list(dict.fromkeys(identificadores))
    if len(identificadores) < 2:
        raise ValueError("Selecciona al menos dos artículos.")

    desconocidos = [identificador for identificador in identificadores if identificador not in articulos]
    if desconocidos:
        raise ValueError(f"IDs que no están en el corpus: {desconocidos}")

    ids_corpus = sorted(articulos)
    todos_los_documentos = [articulos[identificador]["abstract_normalizado"] for identificador in ids_corpus]
    vectores_tfidf = construir_tfidf(todos_los_documentos)
    vector_por_id = dict(zip(ids_corpus, vectores_tfidf))
    resultados = []
    for id_a, id_b in combinations(identificadores, 2):
        articulo_a = articulos[id_a]
        articulo_b = articulos[id_b]
        texto_a = articulo_a["abstract_normalizado"]
        texto_b = articulo_b["abstract_normalizado"]
        medidas = comparar_clasicos(
            texto_a,
            texto_b,
            vector_por_id[id_a],
            vector_por_id[id_b],
        )
        coberturas = {}

        if glove is not None:
            medidas["GloVe (coseno)"] = similitud_embeddings(
                texto_a, texto_b, glove
            )
            cubiertos_a, total_a = cobertura_vocabulario(texto_a, glove)
            cubiertos_b, total_b = cobertura_vocabulario(texto_b, glove)
            coberturas["GloVe cobertura artículo A"] = f"{cubiertos_a}/{total_a}"
            coberturas["GloVe cobertura artículo B"] = f"{cubiertos_b}/{total_b}"
        if fasttext is not None:
            medidas["FastText (coseno)"] = similitud_embeddings(
                texto_a, texto_b, fasttext
            )
            cubiertos_a, total_a = cobertura_vocabulario(texto_a, fasttext)
            cubiertos_b, total_b = cobertura_vocabulario(texto_b, fasttext)
            coberturas["FastText cobertura artículo A"] = f"{cubiertos_a}/{total_a}"
            coberturas["FastText cobertura artículo B"] = f"{cubiertos_b}/{total_b}"

        resultados.append({
            "id_a": id_a,
            "titulo_a": articulo_a.get("titulo", ""),
            "autores_a": articulo_a.get("autores", ""),
            "id_b": id_b,
            "titulo_b": articulo_b.get("titulo", ""),
            "autores_b": articulo_b.get("autores", ""),
            "medidas": medidas,
            "coberturas": coberturas,
        })
    return resultados
