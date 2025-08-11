# src/preprocesamiento_multi_salida.py
import sys, os, glob
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# ==== Columnas según tu CSV ====
COLUMNA_FECHA   = "datetime"
COLUMNA_TEMP    = "temperature"
COLUMNA_HUMEDAD = "humidity"
# ===============================

# Parámetros
FRECUENCIA_OBJETIVO = "1h"          # minúscula para evitar warning
VENTANA = 24
HORIZONTE = 1
TOLERANCIA_NEAREST = "30min"
LIMITE_INTERPOLACION = 3
SALIDA_DIR = "data/processed"

# Features temporales
USAR_HORA_CICLICA = True  # añade hora_sin, hora_cos
USAR_HORA_CICLICA = True
USAR_DIA_SEMANA_CICLICO = True


def resolver_csv():
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
        raise SystemExit(f"❌ Faltan columnas {faltan}. Columnas: {list(df.columns)}")
    df[COLUMNA_FECHA] = pd.to_datetime(df[COLUMNA_FECHA], errors="coerce")
    df = df.dropna(subset=[COLUMNA_FECHA]).sort_values(COLUMNA_FECHA).reset_index(drop=True)
    return df

def alinear_a_rejilla_horaria(df):
    t0, t1 = df[COLUMNA_FECHA].iloc[0], df[COLUMNA_FECHA].iloc[-1]
    inicio, fin = t0.ceil("h"), t1.floor("h")
    if inicio >= fin:
        inicio, fin = t0.round("h"), t1.round("h")

    rejilla = pd.DataFrame({COLUMNA_FECHA: pd.date_range(inicio, fin, freq=FRECUENCIA_OBJETIVO)})
    base = df[[COLUMNA_FECHA, COLUMNA_TEMP, COLUMNA_HUMEDAD]].sort_values(COLUMNA_FECHA)

    # nearest ± tolerancia
    alineado = pd.merge_asof(
        rejilla, base, on=COLUMNA_FECHA, direction="nearest",
        tolerance=pd.Timedelta(TOLERANCIA_NEAREST)
    )
    usados_nearest = alineado[COLUMNA_TEMP].notna().sum()

    alineado = alineado.set_index(COLUMNA_FECHA).sort_index()
    antes = alineado.isna().sum()

    alineado = alineado.interpolate(method="time", limit=LIMITE_INTERPOLACION, limit_direction="both")
    alineado[COLUMNA_HUMEDAD] = alineado[COLUMNA_HUMEDAD].clip(0, 100)
    despues = alineado.isna().sum()

    print(f"⏱️ Rejilla 1h: {len(alineado)} puntos | usados nearest: {usados_nearest}")
    print(f"🧩 NAs antes: {antes.to_dict()} | después (limit={LIMITE_INTERPOLACION}): {despues.to_dict()}")

    return alineado.reset_index()

def agregar_features_temporales(df_h):
    df_h = df_h.copy()
    features = [COLUMNA_TEMP, COLUMNA_HUMEDAD]

    if USAR_HORA_CICLICA:
        horas = df_h[COLUMNA_FECHA].dt.hour.astype(float).to_numpy()
        df_h["hora_sin"] = np.sin(2 * np.pi * horas / 24.0)
        df_h["hora_cos"] = np.cos(2 * np.pi * horas / 24.0)
        features += ["hora_sin", "hora_cos"]

    if USAR_DIA_SEMANA_CICLICO:
        dow = df_h[COLUMNA_FECHA].dt.dayofweek.astype(float).to_numpy()  # 0=Lunes ... 6=Domingo
        df_h["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
        df_h["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)
        features += ["dow_sin", "dow_cos"]

    print(f"🔧 Features de entrada: {features}")
    return df_h, features


def construir_ventanas_y_targets(df_horaria, cols_X):
    valores_X = df_horaria[cols_X].to_numpy(dtype=float)
    valores_y = df_horaria[[COLUMNA_TEMP, COLUMNA_HUMEDAD]].to_numpy(dtype=float)

    X_list, y_list = [], []
    total = len(df_horaria) - VENTANA - HORIZONTE + 1
    if total <= 0:
        raise SystemExit("❌ Muy pocos puntos tras alinear.")
    descartadas = 0

    for i in range(total):
        x_win = valores_X[i:i+VENTANA, :]
        y_tgt = valores_y[i+VENTANA+(HORIZONTE-1), :]
        if np.isnan(x_win).any() or np.isnan(y_tgt).any():
            descartadas += 1
            continue
        X_list.append(x_win)
        y_list.append(y_tgt)

    X = np.array(X_list)
    y = np.array(y_list)
    print(f"📦 Ventanas: {len(X)} (descartadas por NaN: {descartadas}) | n_features={X.shape[2]}")
    return X, y

def split_y_normalizar(X, y):
    # Split temporal 80/20
    n = len(X); corte = int(n * 0.8)
    X_train, X_test = X[:corte], X[corte:]
    y_train, y_test = y[:corte], y[corte:]

    # Normalizar SOLO y por variable (para reportar en unidades reales luego)
    esc_temp, esc_hum = MinMaxScaler(), MinMaxScaler()
    y_train_t = esc_temp.fit_transform(y_train[:, [0]])
    y_train_h = esc_hum.fit_transform(y_train[:, [1]])
    y_test_t  = esc_temp.transform(y_test[:, [0]])
    y_test_h  = esc_hum.transform(y_test[:, [1]])
    y_train_scaled = np.hstack([y_train_t, y_train_h])
    y_test_scaled  = np.hstack([y_test_t,  y_test_h])

    # Normalizar X (todas las features) con StandardScaler
    esc_X = StandardScaler()
    n_feat = X_train.shape[2]
    X_train_2d = X_train.reshape(-1, n_feat)
    X_test_2d  = X_test.reshape(-1, n_feat)

    X_train_scaled = esc_X.fit_transform(X_train_2d).reshape(X_train.shape)
    X_test_scaled  = esc_X.transform(X_test_2d).reshape(X_test.shape)

    return (X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, esc_temp, esc_hum, esc_X)

def main():
    ruta_csv = resolver_csv()
    print(f"📄 Usando CSV: {ruta_csv}")

    df = cargar_dataset(ruta_csv)
    df_h1 = alinear_a_rejilla_horaria(df)
    df_h1, cols_X = agregar_features_temporales(df_h1)
    X, y = construir_ventanas_y_targets(df_h1, cols_X)
    X_train, X_test, y_train, y_test, esc_t, esc_h, esc_X = split_y_normalizar(X, y)

    os.makedirs(SALIDA_DIR, exist_ok=True)
    np.save(os.path.join(SALIDA_DIR, "X_train.npy"), X_train)
    np.save(os.path.join(SALIDA_DIR, "X_test.npy"),  X_test)
    np.save(os.path.join(SALIDA_DIR, "y_train.npy"), y_train)
    np.save(os.path.join(SALIDA_DIR, "y_test.npy"),  y_test)

    # Escaladores
    joblib.dump(esc_t, os.path.join(SALIDA_DIR, "escalador_temperatura.pkl"))
    joblib.dump(esc_h, os.path.join(SALIDA_DIR, "escalador_humedad.pkl"))
    joblib.dump(esc_X, os.path.join(SALIDA_DIR, "escalador_entradas.pkl"))

    print("✅ Preprocesamiento completado.")
    print(f"   X_train: {X_train.shape} | X_test: {X_test.shape}  (X YA VIENE NORMALIZADO)")
    print(f"   y_train: {y_train.shape} | y_test: {y_test.shape}")
    print(f"   Features usadas: {cols_X}")
    print(f"   Guardado en: {SALIDA_DIR}")
    print("   Escaladores: escalador_temperatura.pkl, escalador_humedad.pkl, escalador_entradas.pkl")

if __name__ == "__main__":
    main()
