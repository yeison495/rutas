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


# =============================================================================
# 2. CREACION Y ENTRENAMIENTO DEL MODELO
# =============================================================================
def crear_modelo() -> Pipeline:
    """
    Pipeline = preprocesamiento + modelo en un solo objeto.
    Las variables de texto (ruta, estacion, dia, clima) se convierten a
    columnas 0/1 con One-Hot Encoding, porque la regresion solo entiende numeros.
    """
    preprocesamiento = ColumnTransformer([
        ("categoricas", OneHotEncoder(handle_unknown="ignore"), COLUMNAS_CATEGORICAS),
        ("numericas", "passthrough", COLUMNAS_NUMERICAS),
    ])
    return Pipeline([("preprocesamiento", preprocesamiento),
                     ("regresion", LinearRegression())])


def coeficientes(modelo: Pipeline) -> pd.Series:
    nombres = modelo.named_steps["preprocesamiento"].get_feature_names_out()
    nombres = [n.split("__", 1)[1] for n in nombres]
    return pd.Series(modelo.named_steps["regresion"].coef_, index=nombres)


# =============================================================================
# 5. APLICACION: RUTA MAS RAPIDA (Dijkstra con pesos predichos por el modelo)
# =============================================================================
def ruta_mas_rapida(modelo, origen, destino, dia, hora, clima):
    """
    Cada tramo pesa los minutos que predice el modelo para ese contexto.
    El estado de busqueda es (estacion, ruta actual) para poder sumar la
    penalizacion cuando hay transbordo.
    """
    tramos = list(PARAMETROS_TRAMO)
    entradas = pd.DataFrame([construir_registro(*t, dia, hora, clima) for t in tramos])
    minutos = dict(zip(tramos, modelo.predict(entradas)))

    vecinos = {}
    for ruta, a, b in tramos:
        vecinos.setdefault(a, []).append((b, ruta))

    mejor = {(origen, None): 0.0}
    anterior = {}
    cola = [(0.0, origen, None)]
    while cola:
        costo, estacion, ruta_actual = heapq.heappop(cola)
        if estacion == destino:
            camino, estado = [], (estacion, ruta_actual)
            while estado in anterior:
                previo = anterior[estado]
                camino.append((previo[0], estado[0], estado[1],
                               minutos[(estado[1], previo[0], estado[0])]))
                estado = previo
            return camino[::-1], costo
        if costo > mejor.get((estacion, ruta_actual), float("inf")):
            continue
        for siguiente, ruta in vecinos.get(estacion, []):
            paso = minutos[(ruta, estacion, siguiente)]
            if ruta_actual is not None and ruta != ruta_actual:
                paso += PENALIZACION_TRANSBORDO
            estado = (siguiente, ruta)
            if costo + paso < mejor.get(estado, float("inf")):
                mejor[estado] = costo + paso
                anterior[estado] = (estacion, ruta_actual)
                heapq.heappush(cola, (costo + paso, siguiente, ruta))
    return None, None
    

# =============================================================================
# PROGRAMA PRINCIPAL
# =============================================================================
def main() -> None:
    titulo("EJERCICIO: REGRESION LINEAL - TIEMPO DE VIAJE EN BUS")

    # ---------------------------------------------------------------- 1. Datos
    titulo("1. PREPARACION DE LOS DATOS")
    datos = generar_dataset()
    datos.to_csv("dataset_tiempos_viaje.csv", index=False)
    print(f"Registros: {len(datos)} | Tramos: {len(PARAMETROS_TRAMO)} | "
          f"Rutas: {len(RUTAS_BUS)}")
    print("\nMuestra de los datos (X = contexto del viaje, y = minutos):")
    print(datos.sample(5, random_state=SEMILLA)[
        ["ruta", "estacion_origen", "estacion_destino", "dia_semana",
         "hora_salida", "clima", TARGET]].to_string(index=False))

    X = datos[COLUMNAS_CATEGORICAS + COLUMNAS_NUMERICAS]
    y = datos[TARGET]
    # 80% para que el modelo aprenda, 20% guardado para evaluarlo con datos nuevos
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEMILLA)
    print(f"\nEntrenamiento: {len(X_train)} | Prueba: {len(X_test)}")

    # ---------------------------------------------------------- 2. Entrenamiento
    titulo("2. CREACION Y ENTRENAMIENTO DEL MODELO (SUPERVISADO)")
    modelo = crear_modelo()
    modelo.fit(X_train, y_train)          # el modelo aprende la relacion X -> y
    print("Modelo de Regresion Lineal entrenado exitosamente.")

    coef = coeficientes(modelo)
    efectos = {
        "Hora pico": coef["es_hora_pico"],
        "Lluvia (vs soleado)": coef["clima_lluvia"] - coef["clima_soleado"],
        "Nublado (vs soleado)": coef["clima_nublado"] - coef["clima_soleado"],
        "Fin de semana": coef["es_fin_de_semana"],
    }
    print("\nLo que aprendio el modelo (minutos que suma o resta por tramo):")
    for nombre, valor in efectos.items():
        print(f"  {nombre:<22}{valor:+.2f} min")
    print(f"  Intercepto            {modelo.named_steps['regresion'].intercept_:.2f}")

    y_pred = modelo.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    print("\nEvaluacion con datos de prueba (nunca vistos por el modelo):")
    print(f"  MAE  (error promedio)          : {mae:.2f} min")
    print(f"  RMSE (penaliza errores grandes): {rmse:.2f} min")
    print(f"  R^2  (varianza explicada)      : {r2:.3f}")

    # ------------------------------------------------------------ 3. Inferencia
    titulo("3. PREDICCIONES CON NUEVOS DATOS (INFERENCIA)")
    nuevos_viajes = pd.DataFrame([
        construir_registro("Ruta_3", "Centro", "Zona_Rosa", "Martes", 7, "lluvia"),
        construir_registro("Ruta_3", "Centro", "Zona_Rosa", "Domingo", 11, "soleado"),
        construir_registro("Ruta_1", "Centro", "Parque_Norte", "Viernes", 18, "nublado"),
    ])
    for viaje, minutos in zip(nuevos_viajes.to_dict("records"), modelo.predict(nuevos_viajes)):
        print(f"  {viaje['estacion_origen']} -> {viaje['estacion_destino']:<13}"
              f"{viaje['dia_semana']:<9}{viaje['hora_salida']:>2}:00  "
              f"{viaje['clima']:<8} => {minutos:.1f} min")

    # --------------------------------------------------------- 4. Visualizacion
    titulo("4. VISUALIZACION DE LOS RESULTADOS")
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": "#c3c2b7",
                         "axes.labelcolor": TINTA, "xtick.color": GRIS,
                         "ytick.color": GRIS, "axes.titlecolor": "#0b0b0b"})
    fig, ejes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle("Regresion Lineal: Prediccion del Tiempo de Viaje por Tramo",
                 fontsize=14, fontweight="bold")

    # (a) Real vs predicho: si el modelo es bueno, los puntos caen sobre la diagonal
    ax = ejes[0, 0]
    limites = [y_test.min() - 0.5, y_test.max() + 0.5]
    ax.plot(limites, limites, "--", color=GRIS, linewidth=1.5, label="Prediccion perfecta")
    ax.scatter(y_test, y_pred, color=AZUL, s=36, alpha=0.75,
               edgecolor="white", linewidth=0.8, label="Viajes de prueba")
    ax.set(title=f"Real vs. Predicho  (R² = {r2:.3f})",
           xlabel="Tiempo real (min)", ylabel="Tiempo predicho (min)")
    ax.legend(frameon=False)

    # (b) Errores (residuos): deben centrarse en 0 sin sesgo
    ax = ejes[0, 1]
    residuos = y_test - y_pred
    ax.hist(residuos, bins=20, color=AZUL, edgecolor="white", linewidth=1.5)
    ax.axvline(0, color=TINTA, linewidth=1.5)
    ax.set(title=f"Distribucion del error  (MAE = {mae:.2f} min)",
           xlabel="Error = real - predicho (min)", ylabel="Cantidad de viajes")

    # (c) Curva del modelo a lo largo del dia (equivale a la 'linea de regresion')
    ax = ejes[1, 0]
    horas = list(range(5, 23))
    for clima, color in (("soleado", AZUL), ("lluvia", NARANJA)):
        curva = pd.DataFrame([construir_registro("Ruta_3", "Centro", "Zona_Rosa",
                                                 "Martes", h, clima) for h in horas])
        ax.plot(horas, modelo.predict(curva), color=color, linewidth=2,
                marker="o", markersize=5, label=f"Prediccion ({clima})")
    reales = datos[(datos.estacion_origen == "Centro") & (datos.estacion_destino == "Zona_Rosa")
                   & (datos.es_fin_de_semana == 0)]
    ax.scatter(reales.hora_salida, reales[TARGET], color=GRIS, s=20, alpha=0.6,
               label="Datos reales (entre semana)")
    for inicio, fin in ((6, 8), (17, 19)):
        ax.axvspan(inicio - 0.5, fin + 0.5, color="#e1e0d9", alpha=0.6, linewidth=0)
    ax.text(7, ax.get_ylim()[1] * 0.97, "Hora pico", ha="center", va="top", color=TINTA)
    ax.text(18, ax.get_ylim()[1] * 0.97, "Hora pico", ha="center", va="top", color=TINTA)
    ax.set(title="Tramo Centro -> Zona_Rosa (Ruta_3), dia laboral",
           xlabel="Hora de salida", ylabel="Tiempo (min)", xticks=horas[::2])
    ax.legend(frameon=False, loc="center", bbox_to_anchor=(0.5, 0.5), fontsize=9)

    # (d) Cuanto pesa cada condicion (coeficientes interpretables)
    ax = ejes[1, 1]
    nombres, valores = list(efectos), list(efectos.values())
    barras = ax.barh(nombres, valores, color=AZUL, height=0.5)
    ax.axvline(0, color=TINTA, linewidth=1)
    for barra, valor in zip(barras, valores):
        ax.text(valor + (0.08 if valor >= 0 else -0.08), barra.get_y() + barra.get_height() / 2,
                f"{valor:+.2f} min", va="center", ha="left" if valor >= 0 else "right",
                color=TINTA)
    ax.set_xlim(min(valores) - 1.1, max(valores) + 0.9)
    ax.invert_yaxis()
    ax.set(title="Efecto de cada condicion en el tiempo del tramo",
           xlabel="Minutos que suma (+) o resta (-)")

    for ax in ejes.flat:
        ax.grid(True, color="#e1e0d9", linewidth=0.8)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig("resultados_regresion.png", dpi=150)
    print("Grafica guardada en resultados_regresion.png")

    # ------------------------------------------------------------- 5. Aplicacion
    titulo("5. APLICACION: RUTA MAS RAPIDA SEGUN EL MODELO")
    consultas = [("Terminal_Sur", "Parque_Norte", "Martes", 7, "lluvia"),
                 ("Barrio_Oeste", "Aeropuerto", "Domingo", 11, "soleado"),
                 ("Aeropuerto", "Terminal_Sur", "Lunes", 9, "soleado")]
    for origen, destino, dia, hora, clima in consultas:
        print(f"\n{origen} -> {destino} | {dia} {hora}:00 | {clima}")
        camino, total = ruta_mas_rapida(modelo, origen, destino, dia, hora, clima)
        if camino is None:
            print("  No existe ruta (las rutas son unidireccionales).")
            continue
        ruta_previa = None
        for a, b, ruta, minutos in camino:
            aviso = f"  [transbordo +{PENALIZACION_TRANSBORDO:.0f} min]" \
                if ruta_previa and ruta != ruta_previa else ""
            print(f"  {a} -> {b} ({ruta}) ~{minutos:.1f} min{aviso}")
            ruta_previa = ruta
        print(f"  Tiempo total estimado: {total:.1f} min")

    plt.show()


if __name__ == "__main__":
    main()