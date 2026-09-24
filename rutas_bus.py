"""
Sistema de recomendacion de rutas de bus.

Diseno acordado:
- Cada ruta de bus es una lista ORDENADA de estaciones (rutas unidireccionales,
  sin retorno al punto de partida).
- El sistema construye, a partir de ese diccionario, un GRAFO DIRIGIDO donde
  cada estacion es un nodo y cada par consecutivo dentro de una ruta es un arco.
- Los transbordos NO se modelan aparte: emergen automaticamente cuando una
  estacion aparece en mas de una ruta.
- Criterio de "mejor ruta" = MENOR NUMERO DE PARADAS TOTALES (no se maneja
  tiempo). Por eso el algoritmo de busqueda es BFS: en un grafo
  no ponderado, BFS garantiza el camino con menos saltos.
- Si no existe camino de A a B, el sistema lo reporta explicitamente en vez
  de fallar silenciosamente (posible en un grafo dirigido sin ciclos).
"""

from collections import deque
# ---------------------------------------------------------------------------
# 1. BASE DE CONOCIMIENTO: rutas de bus (estaciones en orden de recorrido)
# ---------------------------------------------------------------------------
rutas_bus = {
    "Ruta_1": ["Terminal_Sur", "Plaza_Mayor", "Universidad", "Centro", "Parque_Norte"],
    "Ruta_2": ["Barrio_Oeste", "Plaza_Mayor", "Hospital", "Mercado", "Terminal_Este"],
    "Ruta_3": ["Terminal_Sur", "Estadio", "Centro", "Zona_Rosa", "Aeropuerto"],
    "Ruta_4": ["Barrio_Oeste", "Universidad", "Zona_Rosa", "Terminal_Este"],
    "Ruta_5": ["Mercado", "Hospital", "Parque_Norte", "Aeropuerto"],
}
# ---------------------------------------------------------------------------
# 2. CONSTRUCCION DEL GRAFO DIRIGIDO A PARTIR DE LAS RUTAS
# ---------------------------------------------------------------------------
def construir_grafo(rutas):
    """
    Convierte el diccionario de rutas en un grafo dirigido de adyacencia.

    Estructura resultante:
        grafo[estacion_origen] = [(estacion_destino, nombre_ruta), ...]

    Se guarda el nombre de la ruta junto a cada arco porque, al final,
    el sistema debe poder decir "toma la Ruta_X" en cada tramo, no solo
    la secuencia de estaciones.
    """
    grafo = {}
    for nombre_ruta, estaciones in rutas.items():
        for i in range(len(estaciones) - 1):
            origen = estaciones[i]
            destino = estaciones[i + 1]
            grafo.setdefault(origen, []).append((destino, nombre_ruta))
            # Aseguramos que el destino exista como nodo aunque no tenga
            # arcos salientes (ej. una terminal final de ruta).
            grafo.setdefault(destino, [])
    return grafo
