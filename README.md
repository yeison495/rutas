# Sistema de Recomendación de Rutas de Bus

Programa en Python que, dada una estación de origen y una de destino, recomienda el recorrido en bus con **menos paradas**, indicando qué ruta tomar en cada tramo y dónde hacer transbordo.

## Explicación breve

La idea del proyecto fue modelar un sistema de transporte sencillo usando **grafos**, que es uno de los temas que vimos en estructuras de datos y que aplica muy bien a este tipo de problemas.

Cada ruta de bus se representa como una lista ordenada de estaciones. A partir de esas listas el programa construye un **grafo dirigido**: cada estación es un nodo y cada par de estaciones consecutivas dentro de una ruta es un arco. Es dirigido porque las rutas son de un solo sentido, es decir, que se pueda ir de A a B no significa que se pueda volver de B a A.

Algo que nos pareció interesante es que los **transbordos no hubo que programarlos aparte**. Como una misma estación puede aparecer en varias rutas, en el grafo queda conectada con arcos de rutas distintas, así que el transbordo sale solo cuando el camino cambia de ruta en esa estación.

Para buscar el mejor camino usamos **BFS (búsqueda en anchura)**. Como el criterio es el menor número de paradas y no el tiempo ni la distancia, todos los arcos pesan lo mismo, y en un grafo sin pesos BFS ya garantiza el camino más corto en número de saltos, con una implementación más simple. Durante la búsqueda se guarda de dónde se llegó a cada estación (un diccionario de "padres"), y al encontrar el destino se reconstruye el camino hacia atrás y se invierte.

Por último, el sistema maneja los casos de error: si una estación no existe o si no hay forma de llegar al destino (algo posible en un grafo dirigido), lo informa con un mensaje en vez de fallar.

## Estructura del código

El archivo `rutas_bus.py` está organizado en secciones:

| Sección | Función | Qué hace |
|---|---|---|
| 1 | `rutas_bus` | Base de conocimiento: las rutas y sus estaciones en orden. |
| 2 | `construir_grafo` | Convierte las rutas en un grafo dirigido (lista de adyacencia) y guarda en cada arco el nombre de la ruta. |
| 3 | `buscar_camino_mas_corto` / `_reconstruir_camino` | Aplica BFS y reconstruye el camino encontrado. |
| 4 | `describir_viaje` | Traduce el camino a instrucciones legibles, numerando las paradas y marcando los transbordos. |
| 5 | `mejor_ruta` | Función principal que une todo el flujo. |
| 6 | `__main__` | Casos de prueba. |

## Rutas de ejemplo

| Ruta | Estaciones |
|---|---|
| Ruta_1 | Terminal_Sur → Plaza_Mayor → Universidad → Centro → Parque_Norte |
| Ruta_2 | Barrio_Oeste → Plaza_Mayor → Hospital → Mercado → Terminal_Este |
| Ruta_3 | Terminal_Sur → Estadio → Centro → Zona_Rosa → Aeropuerto |
| Ruta_4 | Barrio_Oeste → Universidad → Zona_Rosa → Terminal_Este |
| Ruta_5 | Mercado → Hospital → Parque_Norte → Aeropuerto |

## Cómo ejecutarlo

Se necesita Python 3 y no hay que instalar ninguna librería externa (solo se usa `collections.deque` de la librería estándar).

```bash
python rutas_bus.py
```

Para consultar otro recorrido se puede llamar a la función directamente:

```python
from rutas_bus import mejor_ruta, rutas_bus

print(mejor_ruta(rutas_bus, "Barrio_Oeste", "Aeropuerto"))
```

## Ejemplo de salida

```
=== De 'Barrio_Oeste' a 'Aeropuerto' ===
Ruta optima (3 paradas):
  Parada 1: Barrio_Oeste -> Universidad (Ruta_4)
  Parada 2: Universidad -> Zona_Rosa (Ruta_4)
  Parada 3: Zona_Rosa -> Aeropuerto (Ruta_3)  [transbordo en Zona_Rosa]

=== De 'Aeropuerto' a 'Terminal_Sur' ===
No existe una ruta disponible entre 'Aeropuerto' y 'Terminal_Sur' con las rutas actuales.
```

