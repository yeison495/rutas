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
# ---------------------------------------------------------------------------
# 3. BUSQUEDA DEL CAMINO CON MENOS PARADAS (BFS)
# ---------------------------------------------------------------------------
def buscar_camino_mas_corto(grafo, origen, destino):
    """
    BFS clasico sobre el grafo dirigido. Devuelve la lista de tramos
    [(estacion_origen, estacion_destino, ruta), ...] del camino mas corto
    en numero de paradas, o None si no existe camino.
    """
    if origen not in grafo:
        return None, f"La estacion de origen '{origen}' no existe en el sistema."
    if destino not in grafo:
        return None, f"La estacion de destino '{destino}' no existe en el sistema."
    if origen == destino:
        return [], None

    visitados = {origen}
    padre = {}  # padre[estacion] = (estacion_anterior, ruta_usada)
    cola = deque([origen])

    while cola:
        actual = cola.popleft()
        for vecino, ruta in grafo.get(actual, []):
            if vecino not in visitados:
                visitados.add(vecino)
                padre[vecino] = (actual, ruta)
                if vecino == destino:
                    return _reconstruir_camino(padre, origen, destino), None
                cola.append(vecino)

    return None, f"No existe una ruta disponible entre '{origen}' y '{destino}' con las rutas actuales."

def _reconstruir_camino(padre, origen, destino):
    """
    Reconstruye el camino desde 'destino' hacia 'origen' usando el registro
    de padres del BFS, y lo invierte para presentarlo en orden de viaje.
    Devuelve una lista de tramos: [(estacion_origen, estacion_destino, ruta), ...]
    """
    tramos = []
    actual = destino
    while actual != origen:
        anterior, ruta = padre[actual]
        tramos.append((anterior, actual, ruta))
        actual = anterior
    tramos.reverse()
    return tramos
# ---------------------------------------------------------------------------
# 4. TRADUCCION DEL CAMINO A INSTRUCCIONES LEGIBLES + DETECCION DE TRANSBORDOS
# ---------------------------------------------------------------------------
def describir_viaje(tramos, origen, destino):
    """
    Convierte la lista de tramos en un texto legible, marcando los
    transbordos (cuando la ruta de un tramo difiere de la del tramo anterior).
    """
    if tramos == []:
        return f"Ya estas en '{origen}', no se requiere viaje."

    total_paradas = len(tramos)
    lineas = [f"Ruta optima ({total_paradas} parada{'s' if total_paradas != 1 else ''}):"]

    ruta_anterior = None
    for numero_parada, (est_origen, est_destino, ruta) in enumerate(tramos, start=1):
        marca_transbordo = ""
        if ruta_anterior is not None and ruta != ruta_anterior:
            marca_transbordo = f"  [transbordo en {est_origen}]"
        lineas.append(f"  Parada {numero_parada}: {est_origen} -> {est_destino} ({ruta}){marca_transbordo}")
        ruta_anterior = ruta

    return "\n".join(lineas)
# ---------------------------------------------------------------------------
# 5. FUNCION DE ALTO NIVEL: unir todo el flujo
# ---------------------------------------------------------------------------
def mejor_ruta(rutas, origen, destino):
    grafo = construir_grafo(rutas)
    tramos, error = buscar_camino_mas_corto(grafo, origen, destino)
    if error:
        return error
    return describir_viaje(tramos, origen, destino)
# ---------------------------------------------------------------------------
# 6. CASOS DE PRUEBA (segun lo acordado en el diseno)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    casos_de_prueba = [
        ("Terminal_Sur", "Parque_Norte"),   # esperado: directo por Ruta_1
        ("Barrio_Oeste", "Aeropuerto"),     # esperado: con transbordo(s)
        ("Estadio", "Terminal_Este"),       # esperado: con transbordo
        ("Aeropuerto", "Terminal_Sur"),     # esperado: sin ruta (grafo dirigido)
    ]

    for origen, destino in casos_de_prueba:
        print(f"\n=== De '{origen}' a '{destino}' ===")
        print(mejor_ruta(rutas_bus, origen, destino))