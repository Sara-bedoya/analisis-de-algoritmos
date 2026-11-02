"""Comprueba normalización alfabética y eliminación de stopwords."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.preprocessing.limpieza_texto import normalizar_texto, tokenizar_para_frecuencias

texto_prueba = """
Artificial Intelligence (AI) is transform-
ing education. AI supports learning in 2024!
token_con_guion_bajo and the of 99_items.
"""

normalizado = normalizar_texto(texto_prueba)
tokens = tokenizar_para_frecuencias(texto_prueba)

print("Texto original:")
print(texto_prueba)
print("\nTexto normalizado:")
print(normalizado)
print("\nTokens para frecuencias (sin stopwords):")
print(tokens)

assert "2024" not in normalizado
assert "_" not in normalizado
assert "and" not in tokens
assert "the" not in tokens
assert "of" not in tokens
assert "intelligence" in tokens
print("\nPrueba superada correctamente.")
