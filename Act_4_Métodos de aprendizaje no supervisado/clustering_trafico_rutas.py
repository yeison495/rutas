"""
===============================================================================
 DETECCION DE ESTADOS DE TRAFICO EN EL SISTEMA DE BUSES
 Aprendizaje NO supervisado - Clustering con K-Means
===============================================================================
 Diferencia con el ejercicio supervisado:
   - Supervisado    : le damos al modelo la respuesta correcta (y) y aprende a
                      predecirla.
   - No supervisado : NO hay respuesta correcta. El modelo recibe solo
                      mediciones y descubre por si mismo grupos de viajes que
                      se parecen entre si.

 Problema : cada bus reporta (por GPS) su velocidad en el tramo y que tan lleno
            va. Sin decirle al modelo que es "hora pico" ni "congestion",
            queremos que descubra los estados de trafico que existen.
 Algoritmo: K-Means
 Datos    : dataset sintetico generado dentro de este mismo script (630 viajes,
            mismos datos del ejercicio supervisado). No necesita otros archivos.

 Estructura del script:
   1. Preparacion de los datos
   2. Eleccion del numero de grupos (k)
   3. Creacion y entrenamiento del modelo
   4. Interpretacion de los grupos (evaluacion sin etiquetas)
   5. Clasificar nuevas mediciones (inferencia)
   6. Aplicacion a las rutas: tramos que mas se afectan con la congestion
   7. Visualizacion de los resultados
===============================================================================
"""

import random

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# -----------------------------------------------------------------------------
# CONFIGURACION GENERAL
# -----------------------------------------------------------------------------
SEMILLA = 42
RANGO_K = range(2, 8)                                   # cantidades de grupos a probar
MUESTRAS_POR_TRAMO = 35                                 # 18 tramos x 35 = 630 viajes

# Tramos del sistema de buses: (ruta, origen, destino) -> (distancia km, velocidad base km/h)
# Las rutas son unidireccionales; la distancia y velocidad base son supuestos
# razonables para un corredor urbano de bus.
PARAMETROS_TRAMO = {
    ("Ruta_1", "Terminal_Sur", "Plaza_Mayor"): (2.3, 26),
    ("Ruta_1", "Plaza_Mayor", "Universidad"): (1.8, 22),
    ("Ruta_1", "Universidad", "Centro"): (2.1, 20),
    ("Ruta_1", "Centro", "Parque_Norte"): (3.0, 24),
    ("Ruta_2", "Barrio_Oeste", "Plaza_Mayor"): (2.6, 25),
    ("Ruta_2", "Plaza_Mayor", "Hospital"): (1.5, 19),
    ("Ruta_2", "Hospital", "Mercado"): (2.0, 23),
    ("Ruta_2", "Mercado", "Terminal_Este"): (2.8, 27),
    ("Ruta_3", "Terminal_Sur", "Estadio"): (1.9, 28),
    ("Ruta_3", "Estadio", "Centro"): (2.4, 21),
    ("Ruta_3", "Centro", "Zona_Rosa"): (1.7, 18),
    ("Ruta_3", "Zona_Rosa", "Aeropuerto"): (3.5, 30),
    ("Ruta_4", "Barrio_Oeste", "Universidad"): (2.2, 24),
    ("Ruta_4", "Universidad", "Zona_Rosa"): (2.0, 20),
    ("Ruta_4", "Zona_Rosa", "Terminal_Este"): (2.9, 26),
    ("Ruta_5", "Mercado", "Hospital"): (2.0, 23),
    ("Ruta_5", "Hospital", "Parque_Norte"): (2.5, 22),
    ("Ruta_5", "Parque_Norte", "Aeropuerto"): (3.2, 29),
}
DIAS = ["Lunes", "Martes", "Miercoles", "Jueves", "Viernes", "Sabado", "Domingo"]
CLIMAS = ["soleado", "nublado", "lluvia"]
CARACTERISTICAS = ["velocidad_kmh", "ocupacion_pct"]    # lo que mide el GPS del bus

# Nombre de cada grupo, de menor a mayor velocidad promedio
NOMBRES_GRUPOS = {2: ["Congestionado", "Fluido"],
                  3: ["Congestionado", "Normal", "Fluido"]}

# Colores de las graficas (paleta accesible para daltonismo)
AZUL, NARANJA, GRIS, TINTA, REJILLA = "#2a78d6", "#eb6834", "#898781", "#52514e", "#e1e0d9"
COLORES_GRUPO = {"Fluido": AZUL, "Normal": "#1baf7a", "Congestionado": NARANJA}


def titulo(texto: str) -> None:
    print("\n" + "=" * 70)
    print(f" {texto}")
    print("=" * 70)


def es_hora_pico(hora: int) -> bool:
    return 6 <= hora <= 8 or 17 <= hora <= 19


def generar_dataset() -> pd.DataFrame:
    """
    Simula los viajes que reportarian los buses. Es el mismo generador del
    ejercicio supervisado (misma semilla => exactamente los mismos 630 viajes).

    El tiempo depende de la distancia y la velocidad del tramo, y empeora con
    hora pico y lluvia (mejora un poco en fin de semana). La ocupacion es mayor
    en hora pico. Se agrega ruido aleatorio para que no sea una formula exacta.

    Nota: la hora pico y el clima quedan en el dataset solo para INTERPRETAR los
    grupos al final; el modelo de clustering no los usa.
    """
    random.seed(SEMILLA)
    filas = []
    for (ruta, origen, destino), (distancia, velocidad_base) in PARAMETROS_TRAMO.items():
        for _ in range(MUESTRAS_POR_TRAMO):
            dia = random.choice(DIAS)
            hora = random.randint(5, 22)
            clima = random.choices(CLIMAS, weights=[0.65, 0.20, 0.15])[0]
            pico = es_hora_pico(hora)
            fin_de_semana = dia in ("Sabado", "Domingo")

            velocidad = velocidad_base
            if pico:
                velocidad *= random.uniform(0.60, 0.70)   # congestion
            if fin_de_semana:
                velocidad *= random.uniform(1.05, 1.15)   # menos trafico
            if clima == "lluvia":
                velocidad *= random.uniform(0.85, 0.90)   # vias mojadas

            ocupacion = 75 if pico else 40
            ocupacion = max(5, min(100, round(random.gauss(ocupacion, 12), 1)))

            tiempo = distancia / velocidad * 60
            tiempo = max(1.0, tiempo + random.gauss(0, tiempo * 0.08))

            filas.append({
                "id_viaje": len(filas) + 1, "ruta": ruta,
                "estacion_origen": origen, "estacion_destino": destino,
                "distancia_km": distancia, "dia_semana": dia, "hora_salida": hora,
                "es_hora_pico": int(pico), "es_fin_de_semana": int(fin_de_semana),
                "clima": clima, "ocupacion_pct": ocupacion,
                "tiempo_viaje_min": round(tiempo, 2),
            })
    random.shuffle(filas)   # mezclar para que no queden agrupados por tramo
    return pd.DataFrame(filas)


def nombrar_grupos(centroides: pd.DataFrame) -> dict:
    """K-Means numera los grupos al azar (0, 1, ...). Los nombramos segun su velocidad."""
    orden = centroides["velocidad_kmh"].sort_values().index
    nombres = NOMBRES_GRUPOS.get(len(orden), [f"Grupo {i + 1}" for i in range(len(orden))])
    return dict(zip(orden, nombres))


def main() -> None:
    titulo("EJERCICIO: CLUSTERING CON K-MEANS - ESTADOS DE TRAFICO")

    # =========================================================================
    # 1. PREPARACION DE LOS DATOS
    # =========================================================================
    titulo("1. PREPARACION DE LOS DATOS")
    datos = generar_dataset()
    datos.to_csv("dataset_tiempos_viaje.csv", index=False)
    print("Dataset generado y guardado en dataset_tiempos_viaje.csv")

    # Velocidad efectiva del tramo: lo que realmente mediria el GPS del bus
    datos["velocidad_kmh"] = datos["distancia_km"] / (datos["tiempo_viaje_min"] / 60)

    X = datos[CARACTERISTICAS]          # en no supervisado solo hay X: NO existe "y"
    print(f"Viajes: {len(datos)} | Caracteristicas usadas: {CARACTERISTICAS}")
    print("\nMuestra de los datos:")
    print(X.sample(5, random_state=SEMILLA).round(1).to_string())

    # Escalado: velocidad (~10-35 km/h) y ocupacion (0-100 %) tienen escalas
    # distintas. Sin escalar, K-Means daria mas peso a la ocupacion solo por
    # tener numeros mas grandes.
    escalador = StandardScaler()
    X_escalado = escalador.fit_transform(X)
    print("\nDatos escalados (media 0, desviacion 1) para que ambas variables pesen igual.")

    # =========================================================================
    # 2. ELECCION DEL NUMERO DE GRUPOS (k)
    # =========================================================================
    # K-Means necesita que le digamos cuantos grupos buscar. Probamos varios k:
    #  - Inercia  : que tan compactos son los grupos (menor = mejor, pero
    #               siempre baja al subir k; se busca el "codo")
    #  - Silhouette: que tan separados estan los grupos entre si (-1 a 1,
    #               mayor = mejor). Es el criterio de decision.
    titulo("2. ELECCION DEL NUMERO DE GRUPOS (k)")
    inercias, siluetas = [], []
    print(f"{'k':>3}{'Inercia':>12}{'Silhouette':>13}")
    for k in RANGO_K:
        prueba = KMeans(n_clusters=k, n_init=10, random_state=SEMILLA).fit(X_escalado)
        inercias.append(prueba.inertia_)
        siluetas.append(silhouette_score(X_escalado, prueba.labels_))
        print(f"{k:>3}{prueba.inertia_:>12.1f}{siluetas[-1]:>13.3f}")

    k_optimo = list(RANGO_K)[siluetas.index(max(siluetas))]
    print(f"\nk elegido: {k_optimo} (mayor silhouette = grupos mejor separados)")

    # =========================================================================
    # 3. CREACION Y ENTRENAMIENTO DEL MODELO (NO SUPERVISADO)
    # =========================================================================
    titulo("3. CREACION Y ENTRENAMIENTO DEL MODELO (NO SUPERVISADO)")
    modelo = KMeans(n_clusters=k_optimo, n_init=10, random_state=SEMILLA)
    # fit solo recibe X: el modelo agrupa sin conocer ninguna respuesta
    datos["grupo_id"] = modelo.fit_predict(X_escalado)
    print("Modelo K-Means entrenado exitosamente.")

    # Centroides = el "viaje tipico" de cada grupo, devuelto a unidades reales
    centroides = pd.DataFrame(escalador.inverse_transform(modelo.cluster_centers_),
                              columns=CARACTERISTICAS)
    nombres = nombrar_grupos(centroides)
    datos["estado_trafico"] = datos["grupo_id"].map(nombres)
    centroides.index = centroides.index.map(nombres)

    print("\nViaje tipico (centroide) de cada grupo descubierto:")
    for nombre, fila in centroides.sort_values("velocidad_kmh").iterrows():
        cantidad = (datos["estado_trafico"] == nombre).sum()
        print(f"  {nombre:<15} velocidad {fila.velocidad_kmh:5.1f} km/h | "
              f"ocupacion {fila.ocupacion_pct:5.1f} % | {cantidad} viajes")

    # =========================================================================
    # 4. INTERPRETACION DE LOS GRUPOS (evaluacion sin etiquetas)
    # =========================================================================
    # No hay "accuracy" porque no hay respuesta correcta. Lo que si podemos hacer
    # es comparar los grupos con variables que el modelo NUNCA vio (hora pico,
    # clima) para entender que significa cada grupo.
    titulo("4. INTERPRETACION DE LOS GRUPOS")
    tabla = pd.crosstab(datos["estado_trafico"],
                        datos["es_hora_pico"].map({0: "Fuera de pico", 1: "Hora pico"}))
    print("Grupos vs. hora pico (variable que el modelo NO uso):")
    print(tabla.to_string())

    if k_optimo == 2:
        coinciden = ((datos["estado_trafico"] == "Congestionado") ==
                     (datos["es_hora_pico"] == 1)).mean()
        print(f"\nCoincidencia entre 'Congestionado' y hora pico: {coinciden:.1%}")
        print("=> El modelo descubrio la hora pico por si solo, solo mirando "
              "velocidad y ocupacion.")

    lluvia = datos.groupby("estado_trafico")["clima"].apply(lambda c: (c == "lluvia").mean())
    print("\nPorcentaje de viajes con lluvia en cada grupo:")
    for nombre, valor in lluvia.items():
        print(f"  {nombre:<15}{valor:6.1%}")

    # =========================================================================
    # 5. CLASIFICAR NUEVAS MEDICIONES (INFERENCIA)
    # =========================================================================
    titulo("5. CLASIFICAR NUEVAS MEDICIONES (INFERENCIA)")
    nuevas = pd.DataFrame({"velocidad_kmh": [27.0, 14.5, 19.0],
                           "ocupacion_pct": [35.0, 82.0, 60.0]})
    estados = pd.Series(modelo.predict(escalador.transform(nuevas))).map(nombres)
    for (_, medicion), estado in zip(nuevas.iterrows(), estados):
        print(f"  Bus a {medicion.velocidad_kmh:4.1f} km/h con {medicion.ocupacion_pct:4.0f}% "
              f"de ocupacion => {estado}")