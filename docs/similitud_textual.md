# Requerimiento 1: similitud textual

## Estructura

- `src/similarity/corpus.py`: carga y valida los artículos del CSV del corpus.
- `src/similarity/classic.py`: Levenshtein, coseno TF-IDF, Jaccard y Dice.
- `src/similarity/embeddings.py`: carga vectores GloVe/FastText y promedia
  vectores por abstract.
- `src/similarity/service.py`: calcula las medidas para todas las parejas de
  artículos seleccionadas.
- `src/similarity/cli.py`: recibe IDs, rutas a modelos y destino CSV.
- `main.py`: punto de entrada para ejecutar la interfaz.

## Métodos clásicos

1. **Levenshtein normalizado:** distancia de edición a nivel de carácter entre
   abstracts normalizados; se reporta `1 - distancia / longitud_máxima`.
   Complejidad temporal `O(nm)` y memoria `O(min(n,m))`.
2. **Coseno TF-IDF:** frecuencia de término multiplicada por IDF suavizado
   `ln((N+1)/(df+1))+1`; el IDF se calcula con los 20 abstracts. El producto
   punto y las normas se calculan explícitamente sobre vectores dispersos.
3. **Jaccard:** intersección de conjuntos de tokens dividida por su unión.
4. **Dice:** dos veces la intersección de conjuntos dividida por la suma de sus
   tamaños.

Jaccard y Dice ignoran repeticiones. Coseno TF-IDF conserva frecuencia y pondera
los términos por su rareza en el corpus. Levenshtein conserva el orden y compara
caracteres. Las implementaciones no llaman a una biblioteca que resuelva estas
medidas directamente.

## Modelos con embeddings

Ambos métodos reciben vectores preentrenados locales; el proyecto no entrena
con los 20 abstracts porque ese corpus es demasiado pequeño para aprender
representaciones semánticas confiables. Para cada abstract se promedian los
vectores de los tokens cubiertos y se calcula el coseno de los promedios:

- **GloVe:** embeddings globales preentrenados en texto; el modelo de 50
  dimensiones usado como candidato cubre 400 000 términos.
- **FastText:** vectores preentrenados del modelo en texto `.vec`. Este formato
  contiene vectores de vocabulario y no incluye los n-gramas necesarios para
  inferir una palabra fuera de vocabulario; esos términos se omiten.

Los archivos se esperan en `data/models/`, directorio ignorado por Git para no
subir modelos grandes. Antes de usarlos, se debe registrar en el documento final
el origen, idioma, versión y licencia de cada modelo. Las dos medidas de
embeddings también usan coseno, pero sus puntajes pueden ser negativos y no
deben interpretarse como directamente equivalentes a Jaccard/Dice o a las
medidas normalizadas entre 0 y 1.

## Ejecución

Instalar dependencias con `python -m pip install -r requirements.txt`. Después:

```powershell
python main.py --ids 1 2
python main.py --ids 1 2 3 --salida data/resultados_similitud.csv
python main.py --ids 1 2 --glove data/models/glove-wiki-gigaword-50.gz --fasttext data/models/fasttext-wiki-news-subwords-300.gz
```

Se requieren al menos dos IDs válidos. Con `--salida`, la tabla completa de
parejas incluye el cruce de metadatos (`titulo_a`, `autores_a`, `titulo_b`,
`autores_b`), las medidas y la cobertura de vocabulario. Sin rutas a los
modelos se ejecutan solo las cuatro medidas clásicas.
