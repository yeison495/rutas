"""
===============================================================================
 PREDICCION DEL TIEMPO DE VIAJE EN UN SISTEMA DE TRANSPORTE MASIVO (BUS)
 Aprendizaje supervisado - Regresion Lineal Multiple
===============================================================================
 Problema : dado un tramo (estacion A -> estacion B de una ruta) y el contexto
            del viaje (dia, hora, clima), predecir cuantos minutos tarda el bus.
 Tipo     : regresion supervisada
 Target   : tiempo_viaje_min
 Datos    : dataset sintetico (las estaciones del proyecto son ficticias, por lo
            que no existe una fuente real como GTFS para esta red)

 Estructura del script:
   1. Preparacion de los datos
   2. Creacion y entrenamiento del modelo
   3. Predicciones con nuevos datos (inferencia)
   4. Visualizacion de resultados
   5. Aplicacion: ruta mas rapida usando las predicciones del modelo
===============================================================================
"""

import heapq
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# -----------------------------------------------------------------------------
# CONFIGURACION GENERAL
# -----------------------------------------------------------------------------
SEMILLA = 42                     # reproducibilidad: mismos resultados siempre
MUESTRAS_POR_TRAMO = 35          # 18 tramos x 35 = 630 registros
PENALIZACION_TRANSBORDO = 5.0    # supuesto: minutos de espera al cambiar de bus

# Base de conocimiento: rutas de bus (unidireccionales, estaciones en orden)
RUTAS_BUS = {
    "Ruta_1": ["Terminal_Sur", "Plaza_Mayor", "Universidad", "Centro", "Parque_Norte"],
    "Ruta_2": ["Barrio_Oeste", "Plaza_Mayor", "Hospital", "Mercado", "Terminal_Este"],
    "Ruta_3": ["Terminal_Sur", "Estadio", "Centro", "Zona_Rosa", "Aeropuerto"],
    "Ruta_4": ["Barrio_Oeste", "Universidad", "Zona_Rosa", "Terminal_Este"],
    "Ruta_5": ["Mercado", "Hospital", "Parque_Norte", "Aeropuerto"],
}

# Distancia (km) y velocidad base (km/h) de cada tramo -> supuestos del simulador
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

COLUMNAS_CATEGORICAS = ["ruta", "estacion_origen", "estacion_destino", "dia_semana", "clima"]
COLUMNAS_NUMERICAS = ["distancia_km", "hora_salida", "es_hora_pico", "es_fin_de_semana"]
TARGET = "tiempo_viaje_min"

# Colores de las graficas (paleta accesible para daltonismo)
AZUL, NARANJA, GRIS, TINTA = "#2a78d6", "#eb6834", "#898781", "#52514e"


def titulo(texto: str) -> None:
    print("\n" + "=" * 70)
    print(f" {texto}")
    print("=" * 70)


def es_hora_pico(hora: int) -> bool:
    return 6 <= hora <= 8 or 17 <= hora <= 19


def construir_registro(ruta, origen, destino, dia, hora, clima) -> dict:
    """Arma una fila con las variables de entrada (features) de un viaje."""
    distancia, _ = PARAMETROS_TRAMO[(ruta, origen, destino)]
    return {
        "ruta": ruta, "estacion_origen": origen, "estacion_destino": destino,
        "distancia_km": distancia, "dia_semana": dia, "hora_salida": hora,
        "es_hora_pico": int(es_hora_pico(hora)),
        "es_fin_de_semana": int(dia in ("Sabado", "Domingo")),
        "clima": clima,
    }


# =============================================================================
# 1. PREPARACION DE LOS DATOS
# =============================================================================
def generar_dataset() -> pd.DataFrame:
    """
    Simula viajes reales: el tiempo depende de la distancia y la velocidad del
    tramo, y empeora con hora pico y lluvia (mejora un poco en fin de semana).
    Se agrega ruido aleatorio para que no sea una formula exacta.
    """
    random.seed(SEMILLA)
    filas = []
    for (ruta, origen, destino), (distancia, velocidad_base) in PARAMETROS_TRAMO.items():
        for _ in range(MUESTRAS_POR_TRAMO):
            dia = random.choice(DIAS)
            hora = random.randint(5, 22)
            clima = random.choices(CLIMAS, weights=[0.65, 0.20, 0.15])[0]

            velocidad = velocidad_base
            if es_hora_pico(hora):
                velocidad *= random.uniform(0.60, 0.70)   # congestion
            if dia in ("Sabado", "Domingo"):
                velocidad *= random.uniform(1.05, 1.15)   # menos trafico
            if clima == "lluvia":
                velocidad *= random.uniform(0.85, 0.90)   # vias mojadas

            # Ocupacion del bus: se registra en el dataset pero NO se usa como
            # feature, porque al planear un viaje futuro todavia no se conoce.
            ocupacion = 75 if es_hora_pico(hora) else 40
            ocupacion = max(5, min(100, round(random.gauss(ocupacion, 12), 1)))

            tiempo = distancia / velocidad * 60
            tiempo = max(1.0, tiempo + random.gauss(0, tiempo * 0.08))

            fila = {"id_viaje": len(filas) + 1,
                    **construir_registro(ruta, origen, destino, dia, hora, clima),
                    "ocupacion_pct": ocupacion, TARGET: round(tiempo, 2)}
            filas.append(fila)
    random.shuffle(filas)   # mezclar para que no queden agrupados por tramo
    return pd.DataFrame(filas)