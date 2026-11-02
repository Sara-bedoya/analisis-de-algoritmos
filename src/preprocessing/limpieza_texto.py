"""Normalización de abstracts y tokenización léxica para conteos."""

from __future__ import annotations

import logging
import re
import unicodedata

logger = logging.getLogger(__name__)

# Equivalente práctico a nltk.corpus.stopwords.words("english"), sin descargar NLTK.
# Tras tokenizar solo [a-z]+, las contracciones quedan partidas (don't → don, t).
STOPWORDS_INGLES: frozenset[str] = frozenset(
    {
        "a",
        "about",
        "above",
        "after",
        "again",
        "against",
        "ain",
        "all",
        "am",
        "an",
        "and",
        "any",
        "are",
        "aren",
        "arent",
        "as",
        "at",
        "be",
        "because",
        "been",
        "before",
        "being",
        "below",
        "between",
        "both",
        "but",
        "by",
        "can",
        "couldn",
        "couldnt",
        "d",
        "did",
        "didn",
        "didnt",
        "do",
        "does",
        "doesn",
        "doesnt",
        "doing",
        "don",
        "dont",
        "down",
        "during",
        "each",
        "few",
        "for",
        "from",
        "further",
        "had",
        "hadn",
        "hadnt",
        "has",
        "hasn",
        "hasnt",
        "have",
        "haven",
        "havent",
        "having",
        "he",
        "her",
        "here",
        "hers",
        "herself",
        "him",
        "himself",
        "his",
        "how",
        "i",
        "if",
        "in",
        "into",
        "is",
        "isn",
        "isnt",
        "it",
        "its",
        "itself",
        "just",
        "ll",
        "m",
        "ma",
        "me",
        "mightn",
        "mightnt",
        "more",
        "most",
        "mustn",
        "mustnt",
        "my",
        "myself",
        "needn",
        "neednt",
        "no",
        "nor",
        "not",
        "now",
        "o",
        "of",
        "off",
        "on",
        "once",
        "only",
        "or",
        "other",
        "our",
        "ours",
        "ourselves",
        "out",
        "over",
        "own",
        "re",
        "s",
        "same",
        "shan",
        "shant",
        "she",
        "shes",
        "should",
        "shouldn",
        "shouldnt",
        "shouldve",
        "so",
        "some",
        "such",
        "t",
        "than",
        "that",
        "thatll",
        "the",
        "their",
        "theirs",
        "them",
        "themselves",
        "then",
        "there",
        "these",
        "they",
        "this",
        "those",
        "through",
        "to",
        "too",
        "under",
        "until",
        "up",
        "ve",
        "very",
        "was",
        "wasn",
        "wasnt",
        "we",
        "were",
        "weren",
        "werent",
        "what",
        "when",
        "where",
        "which",
        "while",
        "who",
        "whom",
        "why",
        "will",
        "with",
        "won",
        "wont",
        "wouldn",
        "wouldnt",
        "y",
        "you",
        "youd",
        "youll",
        "your",
        "youre",
        "yours",
        "yourself",
        "yourselves",
        "youve",
    }
)

_PATRON_ALFABETICO = re.compile(r"[a-z]+")


def normalizar_texto(texto: str | None) -> str:
    """Convierte un abstract a minúsculas y deja solo palabras alfabéticas.

    Pasos: une cortes de línea, elimina diacríticos, pasa a minúsculas y
    conserva estrictamente tokens `[a-z]+` (sin dígitos ni guiones bajos).
    No elimina stopwords: eso queda para el conteo de frecuencias.

    Args:
        texto: Abstract original o cadena vacía.

    Returns:
        Texto normalizado separado por espacios. Cadena vacía si el
        argumento es nulo o no contiene letras.

    Raises:
        TypeError: Si el valor no es una cadena ni ``None``.
    """
    if texto is None:
        logger.warning("Se recibió texto nulo; se normaliza como cadena vacía.")
        return ""
    if not isinstance(texto, str):
        raise TypeError(f"El texto debe ser str o None, no {type(texto).__name__}.")

    texto = texto.replace("\u00ad", "")
    texto = re.sub(r"(?<=\w)-\s*\n\s*(?=\w)", "", texto)
    texto = unicodedata.normalize("NFKD", texto)
    texto = "".join(
        caracter for caracter in texto if not unicodedata.combining(caracter)
    )
    texto = texto.lower()
    tokens = _PATRON_ALFABETICO.findall(texto)
    return " ".join(tokens)


def eliminar_stopwords(tokens: list[str]) -> list[str]:
    """Filtra palabras vacías del inglés.

    Args:
        tokens: Palabras ya en minúsculas y solo alfabéticas.

    Returns:
        Tokens que no pertenecen a ``STOPWORDS_INGLES``.
    """
    if not tokens:
        return []
    return [token for token in tokens if token not in STOPWORDS_INGLES]


def tokenizar_para_frecuencias(texto: str | None) -> list[str]:
    """Normaliza, extrae tokens alfabéticos y quita stopwords.

    Args:
        texto: Abstract original.

    Returns:
        Lista de términos de contenido, lista para contar frecuencias.
    """
    texto_limpio = normalizar_texto(texto)
    if not texto_limpio:
        return []
    return eliminar_stopwords(texto_limpio.split())
