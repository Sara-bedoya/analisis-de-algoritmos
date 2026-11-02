# Requerimiento 2: agrupamiento jerárquico

## Flujo

1. Leer los 20 abstracts normalizados del catálogo.
2. Representar cada abstract con los mismos pesos TF-IDF que el Requerimiento 1.
3. Calcular la distancia coseno `d(A,B) = 1 - coseno(A,B)` para cada par.
4. Ejecutar los algoritmos aglomerativos *single*, *complete* y *average*.
5. Para cada árbol, evaluar todos los cortes `k = 2..19` con silhouette.
6. Evaluar la fidelidad estructural global de los tres árboles mediante el Coeficiente de Correlación Cofenética (CPCC).
7. Guardar un dendrograma y exportar métricas, fusiones y asignaciones de grupo.

La matriz de distancia se guarda en `output/clustering/distancias_coseno.csv`.
TF-IDF e IDF se calculan sobre los 20 abstracts completos; clustering se basa
únicamente en los abstracts.

## Implementación de los enlaces

Los algoritmos están en `src/clustering/hierarchical.py`. Empiezan con un
cluster por documento. En cada iteración seleccionan la pareja de clusters con
menor distancia, la fusionan y asignan un nuevo ID al cluster. Los IDs iniciales
son `0..19`; la fusión `i` recibe el ID `20+i`. Las reglas de actualización son:

```text
single:   d(A ∪ B, C) = min(d(A,C), d(B,C))
complete: d(A ∪ B, C) = max(d(A,C), d(B,C))
average:  d(A ∪ B, C) = (|A|d(A,C) + |B|d(B,C)) / (|A|+|B|)
```

Enlaces, silhouette y correlación cofenética están implementados desde cero con
estructuras básicas de Python; no se delega a SciPy ni a scikit-learn. La
actualización de distancias aplica estas fórmulas directamente. Para `n`
documentos, buscar la pareja mínima en cada paso cuesta `O(n³)` tiempo total y
la matriz de distancias ocupa `O(n²)` memoria.

## Evaluación interna: Silhouette y Correlación Cofenética (CPCC)

Para responder rigurosamente al requerimiento de *determinar cuál de los algoritmos
produce agrupamientos más coherentes con sustento matemático*, se aplican dos métricas
complementarias:

### 1. Silhouette Promedio (Evaluación de corte plano)

Para cada documento `i`, `a(i)` es su distancia media a los demás elementos de
su grupo. `b(i)` es la menor distancia media de `i` a cualquiera de los otros
grupos. Su silhouette es:

```text
s(i) = (b(i) - a(i)) / max(a(i), b(i))
```

El promedio sobre los 20 documentos mide separación y cohesión: valores cercanos
a 1 indican grupos separados, alrededor de 0 indican fronteras entre grupos y
valores negativos indican que algunos documentos pueden quedar más cerca de
otro grupo.

### 2. Coeficiente de Correlación Cofenética - CPCC (Evaluación estructural global)

El Coeficiente de Correlación Cofenética (Sokal & Rohlf, 1962) evalúa qué tan fielmente
el dendrograma preserva las distancias originales de la matriz de entrada. 

La **distancia cofenética** $C(i,j)$ es la altura en el dendrograma en la que los
artículos $i$ y $j$ se unen por primera vez dentro del mismo cluster. El CPCC ($r$)
es la correlación de Pearson entre las distancias originales por pares $D(i,j)$ y las
distancias cofenéticas $C(i,j)$:

```text
r = Σ [ (D(i,j) - D̄) (C(i,j) - C̄) ] / √[ Σ (D(i,j) - D̄)² · Σ (C(i,j) - C̄)² ]
```
donde la sumatoria abarca todas las parejas $i < j$ ($M = n(n-1)/2 = 190$ pares en el corpus).
Un valor $r \ge 0.8$ indica una excelente preservación topológica, mientras que valores
bajos señalan distorsión introducida por el criterio de enlace (como el encadenamiento
en *single linkage*).

## Resultado del corpus actual y determinación del ganador

| Enlace | Mejor k (Silhouette) | Mejor Silhouette | Correlación Cofenética (CPCC) | Evaluación de coherencia |
| --- | ---: | ---: | ---: | --- |
| **Average** | 2 | 0.103132 | **0.7842** | **Ganador**: Máxima fidelidad y preservación de distancias reales |
| **Complete** | 2 | 0.103132 | 0.6915 | Coherencia intermedia; compacta clusters a costa de elevar distancias |
| **Single** | 2 | 0.103132 | 0.5821 | Menor coherencia; fenómeno de encadenamiento (*chaining*) |

### Lectura crítica:
1. **Empate en Silhouette**: Los tres criterios obtuvieron el mismo mejor valor de Silhouette (`0.103132` con `k=2`), aislando al artículo 14 como outlier y agrupando los otros 19 artículos. Esto demuestra que un corte plano no discrimina la estructura interna de documentos temáticamente homogéneos.
2. **Desempate por CPCC**: El Coeficiente de Correlación Cofenética desempata de forma concluyente. **Average Linkage (UPGMA)** produce los agrupamientos más coherentes y representativos del corpus, al ponderar la distancia promedio entre clusters y evitar los sesgos extremos del mínimo (Single) y del máximo (Complete).

## Dendrogramas y archivos exportados

`src/clustering/visualization.py` dibuja los segmentos del árbol directamente a
partir del historial de fusiones. El CLI `src/clustering/cli.py` escribe en `output/clustering/`:

- `dendrograma_single.png`, `dendrograma_complete.png` y `dendrograma_average.png`.
- `fusiones_<enlace>.csv`, con los nodos, alturas y tamaños de las fusiones.
- `silhouette.csv`, con la puntuación para cada enlace y cada `k`.
- `metricas_coherencia.csv`, con el resumen de Silhouette, CPCC y lectura de cada criterio.
- `agrupaciones_mejores.csv`, con el grupo asignado a cada artículo en el mejor corte.
- `distancias_coseno.csv`, la matriz de distancias TF-IDF de entrada.

Los dendrogramas muestran los IDs numéricos para mantener legibles las hojas;
`agrupaciones_mejores.csv` relaciona cada ID con su título completo.
