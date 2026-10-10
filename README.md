# Sistema Inteligente de Rutas de Bus

Proyecto en Python que modela un sistema de transporte masivo (bus) ficticio y le aplica, paso a paso, distintas técnicas de inteligencia artificial. Cada actividad está en su propia carpeta y reutiliza la misma red de rutas y estaciones:

1. **Búsqueda y sistemas basados en reglas**: encontrar el recorrido con menos paradas.
2. **Aprendizaje supervisado**: predecir cuántos minutos tarda el bus en cada tramo y buscar la ruta más rápida.
3. **Aprendizaje no supervisado**: descubrir, sin etiquetas, los estados de tráfico del sistema.

## Estructura del proyecto

```
rutas/
├── README.md
├── Act_2_Búsqueda y sistemas basados en reglas/
│   └── rutas_bus.py                       # BFS sobre un grafo dirigido de rutas
├── Act_3_Métodos de aprendizaje supervisado/
│   └── prediccion_tiempos_transporte.py   # Regresión lineal + Dijkstra
└── Act_4_Métodos de aprendizaje no supervisado/
    └── clustering_trafico_rutas.py        # K-Means sobre velocidad y ocupación
```

## Red de rutas (común a todas las actividades)

| Ruta | Estaciones |
|---|---|
| Ruta_1 | Terminal_Sur → Plaza_Mayor → Universidad → Centro → Parque_Norte |
| Ruta_2 | Barrio_Oeste → Plaza_Mayor → Hospital → Mercado → Terminal_Este |
| Ruta_3 | Terminal_Sur → Estadio → Centro → Zona_Rosa → Aeropuerto |
| Ruta_4 | Barrio_Oeste → Universidad → Zona_Rosa → Terminal_Este |
| Ruta_5 | Mercado → Hospital → Parque_Norte → Aeropuerto |

Las rutas son de un solo sentido: poder ir de A a B no significa que se pueda volver de B a A.

## Requisitos e instalación

- Python 3.9 o superior (probado con Python 3.11).
- La **Actividad 2** solo usa la librería estándar, así que no hay que instalar nada.
- Las **Actividades 3 y 4** necesitan algunas librerías externas:

```bash
pip install numpy pandas scikit-learn matplotlib
```

Opcionalmente, se pueden instalar dentro de un entorno virtual:

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

pip install numpy pandas scikit-learn matplotlib
```

> **Nota:** los nombres de las carpetas tienen espacios y tildes, así que en la terminal hay que escribir las rutas **entre comillas**.

---

## Actividad 2 · Búsqueda y sistemas basados en reglas

**Archivo:** `Act_2_Búsqueda y sistemas basados en reglas/rutas_bus.py`

Dada una estación de origen y una de destino, recomienda el recorrido con **menos paradas**, indicando qué ruta tomar en cada tramo y dónde hacer transbordo.

### Cómo funciona

Cada ruta es una lista ordenada de estaciones. A partir de esas listas el programa construye un **grafo dirigido**: cada estación es un nodo y cada par de estaciones consecutivas de una ruta es un arco.

Los **transbordos no hubo que programarlos aparte**. Como una misma estación puede aparecer en varias rutas, en el grafo queda conectada con arcos de rutas distintas, y el transbordo aparece solo cuando el camino cambia de ruta en esa estación.

Para buscar el mejor camino se usa **BFS (búsqueda en anchura)**. Como el criterio es el menor número de paradas, todos los arcos pesan lo mismo, y en un grafo sin pesos BFS garantiza el camino más corto en número de saltos. Durante la búsqueda se guarda de dónde se llegó a cada estación (un diccionario de "padres") y, al llegar al destino, se reconstruye el camino hacia atrás. Si una estación no existe o no hay forma de llegar al destino, el programa lo informa con un mensaje en vez de fallar.

| Sección | Función | Qué hace |
|---|---|---|
| 1 | `rutas_bus` | Base de conocimiento: las rutas y sus estaciones en orden. |
| 2 | `construir_grafo` | Convierte las rutas en un grafo dirigido (lista de adyacencia) y guarda en cada arco el nombre de la ruta. |
| 3 | `buscar_camino_mas_corto` / `_reconstruir_camino` | Aplica BFS y reconstruye el camino encontrado. |
| 4 | `describir_viaje` | Traduce el camino a instrucciones legibles, numerando las paradas y marcando los transbordos. |
| 5 | `mejor_ruta` | Función principal que une todo el flujo. |
| 6 | `__main__` | Casos de prueba. |

### Cómo ejecutarlo

```bash
cd "Act_2_Búsqueda y sistemas basados en reglas"
python rutas_bus.py
```

Para consultar otro recorrido se puede llamar a la función directamente (desde esa misma carpeta):

```python
from rutas_bus import mejor_ruta, rutas_bus

print(mejor_ruta(rutas_bus, "Barrio_Oeste", "Aeropuerto"))
```

### Ejemplo de salida

```
=== De 'Barrio_Oeste' a 'Aeropuerto' ===
Ruta optima (3 paradas):
  Parada 1: Barrio_Oeste -> Universidad (Ruta_4)
  Parada 2: Universidad -> Zona_Rosa (Ruta_4)
  Parada 3: Zona_Rosa -> Aeropuerto (Ruta_3)  [transbordo en Zona_Rosa]

=== De 'Aeropuerto' a 'Terminal_Sur' ===
No existe una ruta disponible entre 'Aeropuerto' y 'Terminal_Sur' con las rutas actuales.
```

---

## Actividad 3 · Métodos de aprendizaje supervisado

**Archivo:** `Act_3_Métodos de aprendizaje supervisado/prediccion_tiempos_transporte.py`

Dado un tramo (estación A → estación B de una ruta) y el contexto del viaje (día, hora y clima), **predice cuántos minutos tarda el bus** usando una **regresión lineal múltiple**. Después usa esas predicciones como pesos del grafo para encontrar la **ruta más rápida** con Dijkstra (ya no la de menos paradas, como en la Actividad 2).

Como las estaciones son ficticias, no existe una fuente real de datos (como GTFS) para esta red, así que el script genera un **dataset sintético** de 630 viajes (18 tramos × 35 muestras) con una semilla fija para que los resultados sean reproducibles.

| Sección | Qué hace |
|---|---|
| Configuración | Semilla, rutas, parámetros de cada tramo (distancia, velocidad), días, climas y columnas del modelo. |
| 1. Preparación de los datos | `generar_dataset` crea los viajes sintéticos, los guarda en CSV y los divide en entrenamiento (80 %) y prueba (20 %). |
| 2. Creación y entrenamiento | `crear_modelo` arma un `Pipeline` con `OneHotEncoder` para las variables categóricas y `LinearRegression`. Se evalúa con MAE, RMSE y R². |
| 3. Inferencia | Predice el tiempo de viajes nuevos que el modelo nunca vio. |
| 4. Visualización | Genera 4 gráficas: real vs. predicho, distribución del error, tiempo del tramo Centro → Zona_Rosa a lo largo del día y efecto de cada condición (coeficientes). |
| 5. Aplicación | `ruta_mas_rapida` aplica Dijkstra usando como peso de cada tramo el tiempo predicho, sumando 5 min de penalización por transbordo. |

**Archivos que genera** (en la carpeta desde donde se ejecuta):

- `dataset_tiempos_viaje.csv`: el dataset sintético.
- `resultados_regresion.png`: las gráficas del modelo.

### Cómo ejecutarlo

```bash
cd "Act_3_Métodos de aprendizaje supervisado"
python prediccion_tiempos_transporte.py
```

Al final se abre una ventana con las gráficas; el script termina al cerrarla.

### Ejemplo de salida (resumido)

```
Evaluacion con datos de prueba (nunca vistos por el modelo):
  MAE  (error promedio)          : 0.51 min
  RMSE (penaliza errores grandes): 0.64 min
  R^2  (varianza explicada)      : 0.887

Terminal_Sur -> Parque_Norte | Martes 7:00 | lluvia
  Terminal_Sur -> Estadio (Ruta_3) ~7.7 min
  Estadio -> Centro (Ruta_3) ~10.9 min
  Centro -> Parque_Norte (Ruta_1) ~11.8 min  [transbordo +5 min]
  Tiempo total estimado: 35.3 min
```

---

## Actividad 4 · Métodos de aprendizaje no supervisado

**Archivo:** `Act_4_Métodos de aprendizaje no supervisado/clustering_trafico_rutas.py`

Cada bus reporta por GPS su **velocidad** en el tramo y **qué tan lleno va**. Sin decirle al modelo qué es "hora pico" ni "congestión", el algoritmo **K-Means** agrupa los viajes que se parecen entre sí y descubre por su cuenta los estados de tráfico del sistema.

A diferencia de la Actividad 3, aquí **no hay una respuesta correcta (y)**: el modelo solo recibe las mediciones. El script genera los mismos 630 viajes sintéticos de la actividad supervisada, así que no necesita ningún archivo externo.

| Sección | Qué hace |
|---|---|
| 1. Preparación de los datos | Genera el dataset, toma `velocidad_kmh` y `ocupacion_pct` y las escala con `StandardScaler` para que pesen igual. |
| 2. Elección de k | Prueba k = 2…7 y compara la inercia (método del codo) y el coeficiente silhouette; elige el k con mayor silhouette. |
| 3. Creación y entrenamiento | Entrena `KMeans` y muestra el centroide (viaje típico) de cada grupo, nombrándolos como Fluido / Normal / Congestionado. |
| 4. Interpretación | Compara los grupos con variables que el modelo **no** usó (hora pico, lluvia) para comprobar que tienen sentido. |
| 5. Inferencia | Clasifica mediciones nuevas de buses en uno de los grupos. |
| 6. Aplicación a las rutas | Compara el tiempo promedio de cada tramo en estado fluido vs. congestionado para encontrar los tramos más afectados. |
| 7. Visualización | Genera las gráficas (codo, silhouette, dispersión de los grupos y tramos más afectados). |

**Archivos que genera** (en la carpeta desde donde se ejecuta):

- `dataset_tiempos_viaje.csv`: el dataset sintético.
- `resultados_clustering.png`: las gráficas del clustering.

### Cómo ejecutarlo

```bash
cd "Act_4_Métodos de aprendizaje no supervisado"
python clustering_trafico_rutas.py
```

Al final se abre una ventana con las gráficas; el script termina al cerrarla.

### Ejemplo de salida (resumido)

```
k elegido: 2 (mayor silhouette = grupos mejor separados)

Viaje tipico (centroide) de cada grupo descubierto:
  Congestionado   velocidad  15.9 km/h | ocupacion  74.1 % | 239 viajes
  Fluido          velocidad  24.5 km/h | ocupacion  39.6 % | 391 viajes

Coincidencia entre 'Congestionado' y hora pico: 95.2%
=> El modelo descubrio la hora pico por si solo, solo mirando velocidad y ocupacion.

  Ruta    Tramo                            Fluido  Congestionado   Extra
  Ruta_5  Parque_Norte -> Aeropuerto        6.5 m         10.5 m   +3.9
  Ruta_4  Zona_Rosa -> Terminal_Este        6.3 m         10.1 m   +3.8
```

---

## Ejecutar todo de una vez

Desde la raíz del repositorio:

```bash
python "Act_2_Búsqueda y sistemas basados en reglas/rutas_bus.py"
python "Act_3_Métodos de aprendizaje supervisado/prediccion_tiempos_transporte.py"
python "Act_4_Métodos de aprendizaje no supervisado/clustering_trafico_rutas.py"
```

Hay que tener en cuenta que, ejecutados así, los archivos `.csv` y `.png` se guardan en la raíz y no dentro de cada carpeta.
