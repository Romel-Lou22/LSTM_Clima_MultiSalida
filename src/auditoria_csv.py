# src/auditoria_csv.py
import sys, glob
import pandas as pd
import numpy as np
from collections import Counter


COLUMNA_FECHA   = "datetime"   
COLUMNA_TEMP    = "temperature"      
COLUMNA_HUMEDAD = "humidity"     
# ===========================================================

def inferir_frecuencia(deltas):
    segundos = [int(d.total_seconds()) for d in deltas if pd.notna(d)]
    if not segundos:
        return None, None
    moda_s = Counter(segundos).most_common(1)[0][0]
    mediana_s = int(np.median(segundos))

    def a_texto(s):
        if s % 86400 == 0: return f"{s//86400}D"
        if s % 3600  == 0: return f"{s//3600}H"
        if s % 60    == 0: return f"{s//60}min"
        return f"{s}s"

    return a_texto(moda_s), a_texto(mediana_s)

def main():
    # 1) Resolver ruta del CSV
    RUTA_CSV_FIJA = r"C:\Users\ZenBook\Desktop\redLSTM\LSTM_Clima\data\openweather_data.csv" 
    ruta_csv = sys.argv[1] if len(sys.argv) > 1 else RUTA_CSV_FIJA
    if ruta_csv is None:
        archivos = glob.glob("*.csv")
        if not archivos:
            print("❌ No encontré archivos .csv en la raíz. Uso: python src/auditoria_csv.py TU_ARCHIVO.csv")
            sys.exit(1)
        ruta_csv = archivos[0]

    # 2) Cargar y validar columnas
    df = pd.read_csv(ruta_csv)
    columnas = set(df.columns)

    if COLUMNA_FECHA not in columnas:
        print(f"❌ No encuentro la columna de tiempo '{COLUMNA_FECHA}'. Columnas: {list(df.columns)}")
        sys.exit(1)

    df[COLUMNA_FECHA] = pd.to_datetime(df[COLUMNA_FECHA], errors="coerce")
    df = df.sort_values(COLUMNA_FECHA).reset_index(drop=True)

    # 3) Resumen base
    duplicados = df.duplicated(COLUMNA_FECHA).sum()
    n_nat = df[COLUMNA_FECHA].isna().sum()
    print(f"📄 Archivo: {ruta_csv}")
    print(f"🔢 Filas: {len(df)} | ⏱️ Duplicados en tiempo: {duplicados} | NaT (fechas inválidas): {n_nat}")

    # 4) Frecuencia temporal
    deltas = df[COLUMNA_FECHA].diff().dropna()
    freq_moda, freq_mediana = inferir_frecuencia(deltas)
    print(f"⏳ Frecuencia estimada → moda: {freq_moda} | mediana: {freq_mediana}")

    # 5) Huecos vs. rejilla regular (si se pudo inferir)
    freq = freq_moda or freq_mediana
    if freq:
        idx = pd.date_range(df[COLUMNA_FECHA].iloc[0], df[COLUMNA_FECHA].iloc[-1], freq=freq)
        huecos = len(idx.difference(df[COLUMNA_FECHA]))
        print(f"🕳️ Huecos estimados respecto a rejilla {freq}: {huecos}")

    # 6) Estadísticas básicas
    for col, nombre in [(COLUMNA_TEMP, "temperatura"), (COLUMNA_HUMEDAD, "humedad")]:
        if col in columnas:
            s = df[col].dropna()
            print(f"📈 {nombre}: min={s.min():.3f} | max={s.max():.3f} | media={s.mean():.3f} | std={s.std():.3f} | nNA={df[col].isna().sum()}")
        else:
            print(f"⚠️ No encuentro columna '{col}'. Ajusta COLUMNA_TEMP/COLUMNA_HUMEDAD si tus nombres son distintos.")

    # 7) Sugerencia de ventana (WINDOW)
    sugerencias = {"15min":96, "30min":48, "60min":24, "1H":24, "24H":7, "1D":7}
    ventana = sugerencias.get(freq, None)
    print(f"🪄 Ventana sugerida (WINDOW): {ventana if ventana else 'revisar según frecuencia'} | Horizonte sugerido: 1 paso (t+1)")

if __name__ == "__main__":
    main()
