# src/preprocesamiento_multi_salida.py
import sys, os, glob
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler

# ==== Configura aquí si tus columnas se llaman distinto ====
COLUMNA_FECHA   = "datetime"      # p.ej. "fecha", "timestamp"
COLUMNA_TEMP    = "temperature"   # p.ej. "temp", "temperatura"
COLUMNA_HUMEDAD = "humidity"      # p.ej. "humid", "humedad"
# ===========================================================

# Parámetros del pipeline
FRECUENCIA_OBJETIVO = "1h"        # rejilla horaria exacta
VENTANA = 24                      # WINDOW: 24 horas de contexto
HORIZONTE = 1                     # predecir t+1
TOLERANCIA_NEAREST = "30min"      # usar dato más cercano si cae dentro de ±30 min
LIMITE_INTERPOLACION = 3          # máximo de 3 horas consecutivas a interpolar
SALIDA_DIR = "data/processed"     # a dónde guardar .npy/.pkl

def resolver_csv():
    """1) CLI arg > 2) env CSV_PATH > 3) primer .csv en la raíz."""
    RUTA_CSV_FIJA = r"C:\Users\ZenBook\Desktop\redLSTM\LSTM_Clima\data\openweather_data.csv" 
    ruta = sys.argv[1] if len(sys.argv) > 1 else RUTA_CSV_FIJA
    if ruta is None:
        archivos = glob.glob("*.csv")
        if not archivos:
            raise SystemExit("❌ No encontré .csv. Usa: python src/preprocesamiento_multi_salida.py TU_ARCHIVO.csv")
        ruta = archivos[0]
    return ruta

def cargar_dataset(ruta_csv):
    df = pd.read_csv(ruta_csv)
    faltan = [c for c in [COLUMNA_FECHA, COLUMNA_TEMP, COLUMNA_HUMEDAD] if c not in df.columns]
    if faltan:
        raise SystemExit(f"❌ Faltan columnas {faltan}. Columnas disponibles: {list(df.columns)}")
    df[COLUMNA_FECHA] = pd.to_datetime(df[COLUMNA_FECHA], errors="coerce")
    df = df.dropna(subset=[COLUMNA_FECHA]).sort_values(COLUMNA_FECHA).reset_index(drop=True)
    return df

def alinear_a_rejilla_horaria(df):
    """Rejilla exacta 1H: dato más cercano con tolerancia ±30min; luego interpolación temporal limitada."""
    t0 = df[COLUMNA_FECHA].iloc[0]
    t1 = df[COLUMNA_FECHA].iloc[-1]
    inicio = t0.ceil("h")   # arranca en la próxima hora cerrada
    fin    = t1.floor("h")  # termina en la hora cerrada previa

    if inicio >= fin:
        # fallback: usa bordes redondeados por si el rango es muy corto
        inicio = t0.round("h")
        fin    = t1.round("h")

    rejilla = pd.DataFrame({COLUMNA_FECHA: pd.date_range(inicio, fin, freq=FRECUENCIA_OBJETIVO)})
    df = df[[COLUMNA_FECHA, COLUMNA_TEMP, COLUMNA_HUMEDAD]].sort_values(COLUMNA_FECHA)

    # 1) Emparejar por "nearest" con tolerancia
    alineado = pd.merge_asof(
        rejilla, df, on=COLUMNA_FECHA, direction="nearest",
        tolerance=pd.Timedelta(TOLERANCIA_NEAREST)
    )

    usados_nearest = alineado[COLUMNA_TEMP].notna().sum()

    # 2) Interpolar en el tiempo lo que falte (máx. LIMITE_INTERPOLACION horas seguidas)
    alineado = alineado.set_index(COLUMNA_FECHA).sort_index()
    antes_interpol = alineado.isna().sum()

    alineado = alineado.interpolate(method="time", limit=LIMITE_INTERPOLACION, limit_direction="both")

    # 3) Seguridad: humedad en [0,100]
    alineado[COLUMNA_HUMEDAD] = alineado[COLUMNA_HUMEDAD].clip(0, 100)

    despues_interpol = alineado.isna().sum()

    print(f"⏱️ Rejilla 1H: {len(alineado)} puntos | usados por 'nearest': {usados_nearest}")
    print(f"🧩 NAs antes de interpolar: {antes_interpol.to_dict()}")
    print(f"🧩 NAs después de interpolar (limit={LIMITE_INTERPOLACION}): {despues_interpol.to_dict()}")

    return alineado.reset_index()

def construir_ventanas_y_targets(df_horaria):
    """Construye X (VENTANA, 2) y y (2,) y descarta ventanas con NaN."""
    valores = df_horaria[[COLUMNA_TEMP, COLUMNA_HUMEDAD]].to_numpy(dtype=float)
    X_list, y_list = [], []

    total = len(valores) - VENTANA - HORIZONTE + 1
    if total <= 0:
        raise SystemExit("❌ Muy pocos puntos tras alinear. Ajusta fechas o parámetros.")

    descartadas = 0
    for i in range(total):
        x_win = valores[i:i+VENTANA, :]
        y_tgt = valores[i+VENTANA+(HORIZONTE-1), :]
        if np.isnan(x_win).any() or np.isnan(y_tgt).any():
            descartadas += 1
            continue
        X_list.append(x_win)
        y_list.append(y_tgt)

    X = np.array(X_list)
    y = np.array(y_list)
    print(f"📦 Ventanas construidas: {len(X)} (descartadas por NaN: {descartadas})")
    return X, y

def split_temporal_y_normalizar_y(X, y):
    """80/20 temporal. Normaliza SOLO las salidas por variable y guarda escaladores."""
    n = len(X)
    corte = int(n * 0.8)
    X_train, X_test = X[:corte], X[corte:]
    y_train, y_test = y[:corte], y[corte:]

    esc_temp  = MinMaxScaler()
    esc_humid = MinMaxScaler()

    y_train_temp_esc  = esc_temp.fit_transform(y_train[:, [0]])
    y_train_humid_esc = esc_humid.fit_transform(y_train[:, [1]])
    y_test_temp_esc   = esc_temp.transform(y_test[:, [0]])
    y_test_humid_esc  = esc_humid.transform(y_test[:, [1]])

    y_train_esc = np.hstack([y_train_temp_esc, y_train_humid_esc])
    y_test_esc  = np.hstack([y_test_temp_esc,  y_test_humid_esc])

    return (X_train, X_test, y_train_esc, y_test_esc, esc_temp, esc_humid)

def main():
    ruta_csv = resolver_csv()
    print(f"📄 Usando CSV: {ruta_csv}")

    df = cargar_dataset(ruta_csv)
    df_h1 = alinear_a_rejilla_horaria(df)
    X, y = construir_ventanas_y_targets(df_h1)
    X_train, X_test, y_train, y_test, esc_temp, esc_humid = split_temporal_y_normalizar_y(X, y)

    os.makedirs(SALIDA_DIR, exist_ok=True)
    np.save(os.path.join(SALIDA_DIR, "X_train.npy"), X_train)
    np.save(os.path.join(SALIDA_DIR, "X_test.npy"),  X_test)
    np.save(os.path.join(SALIDA_DIR, "y_train.npy"), y_train)
    np.save(os.path.join(SALIDA_DIR, "y_test.npy"),  y_test)

    # Escaladores de SALIDAS (en español)
    joblib.dump(esc_temp,  os.path.join(SALIDA_DIR, "escalador_temperatura.pkl"))
    joblib.dump(esc_humid, os.path.join(SALIDA_DIR, "escalador_humedad.pkl"))

    print("✅ Preprocesamiento completado.")
    print(f"   X_train: {X_train.shape} | X_test: {X_test.shape}")
    print(f"   y_train: {y_train.shape} | y_test: {y_test.shape}")
    print(f"   Guardado en: {SALIDA_DIR}")
    print("   Escaladores: escalador_temperatura.pkl, escalador_humedad.pkl")

if __name__ == "__main__":
    main()
