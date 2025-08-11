# src/entrenamiento_multi_salida.py
import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
import joblib

# =========================
# Parámetros principales
# =========================
DATA_DIR   = "data/processed"
MODELS_DIR = "models"
FIGS_DIR   = "figures"

EPOCHS = 120
BATCH_SIZE = 64
VAL_FRACCION_DENTRO_TRAIN = 0.2  # 20% del TRAIN como validación temporal
PAC_EARLYSTOP = 12
PAC_REDUCELR  = 6
LR_MIN = 1e-5
LR_INICIAL = 1e-3
SEED = 42

# Ponderación de pérdidas (empujar un poco la temperatura)
LOSS_WEIGHTS = {"temperatura": 1.5, "humedad": 1.0}

def set_seeds(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def cargar_datos(directorio):
    X_train = np.load(os.path.join(directorio, "X_train.npy")).astype("float32")
    X_test  = np.load(os.path.join(directorio, "X_test.npy")).astype("float32")
    y_train = np.load(os.path.join(directorio, "y_train.npy")).astype("float32")
    y_test  = np.load(os.path.join(directorio, "y_test.npy")).astype("float32")

    if y_train.ndim != 2 or y_train.shape[1] != 2:
        raise ValueError(f"y_train debe ser (muestras,2). Recibido: {y_train.shape}")
    if y_test.ndim != 2 or y_test.shape[1] != 2:
        raise ValueError(f"y_test debe ser (muestras,2). Recibido: {y_test.shape}")

    y_train_temp, y_train_hum = y_train[:, 0], y_train[:, 1]
    y_test_temp,  y_test_hum  = y_test[:, 0],  y_test[:, 1]
    return X_train, X_test, y_train_temp, y_train_hum, y_test_temp, y_test_hum

def cargar_escaladores(directorio):
    esc_t = joblib.load(os.path.join(directorio, "escalador_temperatura.pkl"))
    esc_h = joblib.load(os.path.join(directorio, "escalador_humedad.pkl"))
    print("🔄 Escaladores de salida cargados.")
    return esc_t, esc_h

def construir_modelo(n_steps, n_features):
    inputs = tf.keras.layers.Input(shape=(n_steps, n_features))
    x = tf.keras.layers.LSTM(64, return_sequences=True, name="lstm_1")(inputs)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.LSTM(32, name="lstm_2")(x)
    x = tf.keras.layers.Dropout(0.2)(x)

    # Rama temperatura
    t = tf.keras.layers.Dense(32, activation="relu", name="temp_dense_1")(x)
    t = tf.keras.layers.Dropout(0.2)(t)
    t = tf.keras.layers.Dense(16, activation="relu", name="temp_dense_2")(t)
    out_temp = tf.keras.layers.Dense(1, activation="linear", name="temperatura")(t)

    # Rama humedad
    h = tf.keras.layers.Dense(32, activation="relu", name="hum_dense_1")(x)
    h = tf.keras.layers.Dropout(0.2)(h)
    h = tf.keras.layers.Dense(16, activation="relu", name="hum_dense_2")(h)
    out_hum = tf.keras.layers.Dense(1, activation="linear", name="humedad")(h)

    model = tf.keras.Model(inputs=inputs, outputs=[out_temp, out_hum], name="LSTM_MultiSalida")
    opt = tf.keras.optimizers.Adam(learning_rate=LR_INICIAL)
    model.compile(
        optimizer=opt,
        loss={"temperatura": "huber", "humedad": "huber"},
        loss_weights=LOSS_WEIGHTS,
        metrics={"temperatura": ["mse", "mae"], "humedad": ["mse", "mae"]}
    )
    return model

def split_validacion_temporal(X_trn, y_t_trn, y_h_trn, frac_val=0.2):
    n = X_trn.shape[0]
    corte = int(n * (1 - frac_val))
    X_tr, X_val = X_trn[:corte], X_trn[corte:]
    y_tr = {"temperatura": y_t_trn[:corte], "humedad": y_h_trn[:corte]}
    y_val = {"temperatura": y_t_trn[corte:],  "humedad": y_h_trn[corte:]}
    return (X_tr, y_tr), (X_val, y_val)

def entrenar(model, X_train, y_train_temp, y_train_hum):
    (X_tr, y_tr), (X_val, y_val) = split_validacion_temporal(X_train, y_train_temp, y_train_hum, VAL_FRACCION_DENTRO_TRAIN)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=PAC_EARLYSTOP, restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=PAC_REDUCELR, min_lr=LR_MIN, verbose=1),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(MODELS_DIR, "mejor_modelo_multi_salida.keras"),
            monitor="val_loss", save_best_only=True, verbose=1
        )
    ]

    hist = model.fit(
        X_tr, y_tr,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        shuffle=False,
        callbacks=callbacks,
        verbose=1
    )
    return hist

def evaluar(model, X_test, y_test_temp, y_test_hum, esc_t=None, esc_h=None):
    y_pred_t, y_pred_h = model.predict(X_test, verbose=0)
    y_pred_t = y_pred_t.ravel()
    y_pred_h = y_pred_h.ravel()

    y_true_t, y_true_h = y_test_temp, y_test_hum
    y_hat_t,  y_hat_h  = y_pred_t,  y_pred_h
    unidades = "(normalizado)"

    if esc_t is not None and esc_h is not None:
        y_true_t = esc_t.inverse_transform(y_true_t.reshape(-1,1)).ravel()
        y_hat_t  = esc_t.inverse_transform(y_hat_t.reshape(-1,1)).ravel()
        y_true_h = esc_h.inverse_transform(y_true_h.reshape(-1,1)).ravel()
        y_hat_h  = esc_h.inverse_transform(y_hat_h.reshape(-1,1)).ravel()
        unidades = "(°C y %HR)"

    t_mse = mean_squared_error(y_true_t, y_hat_t); t_rmse = np.sqrt(t_mse); t_mae = mean_absolute_error(y_true_t, y_hat_t); t_r2 = r2_score(y_true_t, y_hat_t)
    h_mse = mean_squared_error(y_true_h, y_hat_h); h_rmse = np.sqrt(h_mse); h_mae = mean_absolute_error(y_true_h, y_hat_h); h_r2 = r2_score(y_true_h, y_hat_h)


    print(f"\n📊 MÉTRICAS TEST {unidades}")
    print(f"  🌡️ Temperatura → R²={t_r2:.4f} | RMSE={t_rmse:.3f} | MAE={t_mae:.3f} | MSE={t_mse:.4f}")  # ← añade MSE
    print(f"  💧 Humedad     → R²={h_r2:.4f} | RMSE={h_rmse:.3f} | MAE={h_mae:.3f} | MSE={h_mse:.4f}")  # ← añade MSE

    return (y_hat_t, y_hat_h), (y_true_t, y_true_h), {
        "temp":  {"r2": t_r2, "rmse": t_rmse, "mae": t_mae, "mse": t_mse},
        "humid": {"r2": h_r2, "rmse": h_rmse, "mae": h_mae, "mse": h_mse}
    }

# ========== Gráficas ==========
def graficar_historial(hist):
    os.makedirs(FIGS_DIR, exist_ok=True)
    plt.figure(figsize=(11,5))
    plt.plot(hist.history["loss"], label="Pérdida total (train)")
    plt.plot(hist.history["val_loss"], label="Pérdida total (val)")
    plt.xlabel("Época"); plt.ylabel("Loss"); plt.legend(); plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGS_DIR, "historial_entrenamiento.png"), dpi=200)

def graficar_metricas(hist):
    """Figura 2x2: MAE/MSE para temperatura y humedad (train vs val)."""
    os.makedirs(FIGS_DIR, exist_ok=True)
    keys = hist.history.keys()

    def get(k):  # helper por si cambia el nombre
        return hist.history.get(k, None)

    tt_mae, vt_mae = get("temperatura_mae"), get("val_temperatura_mae")
    th_mae, vh_mae = get("humedad_mae"),     get("val_humedad_mae")
    tt_mse, vt_mse = get("temperatura_mse"), get("val_temperatura_mse")
    th_mse, vh_mse = get("humedad_mse"),     get("val_humedad_mse")

    plt.figure(figsize=(12,8))
    # Temp MAE
    plt.subplot(2,2,1)
    plt.plot(tt_mae, label="Train MAE Temp")
    plt.plot(vt_mae, "--", label="Val MAE Temp")
    plt.ylabel("MAE"); plt.title("Temperatura - MAE"); plt.grid(True, alpha=0.3); plt.legend()

    # Temp MSE
    plt.subplot(2,2,2)
    plt.plot(tt_mse, label="Train MSE Temp")
    plt.plot(vt_mse, "--", label="Val MSE Temp")
    plt.ylabel("MSE"); plt.title("Temperatura - MSE"); plt.grid(True, alpha=0.3); plt.legend()

    # Hum MAE
    plt.subplot(2,2,3)
    plt.plot(th_mae, label="Train MAE Humidity")
    plt.plot(vh_mae, "--", label="Val MAE Humidity")
    plt.ylabel("MAE"); plt.title("Humedad - MAE"); plt.grid(True, alpha=0.3); plt.legend()

    # Hum MSE
    plt.subplot(2,2,4)
    plt.plot(th_mse, label="Train MSE Humidity")
    plt.plot(vh_mse, "--", label="Val MSE Humidity")
    plt.ylabel("MSE"); plt.title("Humedad - MSE"); plt.grid(True, alpha=0.3); plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(FIGS_DIR, "metricas_mae_mse.png"), dpi=200)

def graficar_predicciones(y_true_t, y_true_h, y_hat_t, y_hat_h):
    os.makedirs(FIGS_DIR, exist_ok=True)
    n = len(y_true_t)
    idx = np.linspace(0, n-1, min(500, n), dtype=int)

    # Serie temperatura
    plt.figure(figsize=(12,4))
    plt.plot(idx, y_true_t[idx], label="Real")
    plt.plot(idx, y_hat_t[idx], label="Predicho", linestyle="--")
    plt.title("Serie: Temperatura (subconjunto)"); plt.legend(); plt.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(FIGS_DIR, "serie_temperatura.png"), dpi=200)

    # Serie humedad
    plt.figure(figsize=(12,4))
    plt.plot(idx, y_true_h[idx], label="Real")
    plt.plot(idx, y_hat_h[idx], label="Predicho", linestyle="--")
    plt.title("Serie: Humedad (subconjunto)"); plt.legend(); plt.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(FIGS_DIR, "serie_humedad.png"), dpi=200)

    # Dispersión temperatura
    plt.figure(figsize=(5,5))
    min_t, max_t = min(y_true_t.min(), y_hat_t.min()), max(y_true_t.max(), y_hat_t.max())
    plt.scatter(y_true_t, y_hat_t, s=12, alpha=0.6)
    plt.plot([min_t, max_t], [min_t, max_t], "r--", linewidth=1)
    plt.xlabel("Real"); plt.ylabel("Predicho"); plt.title("Real vs Predicho (Temperatura)")
    plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(FIGS_DIR, "scatter_temperatura.png"), dpi=200)

    # Dispersión humedad
    plt.figure(figsize=(5,5))
    min_h, max_h = min(y_true_h.min(), y_hat_h.min()), max(y_true_h.max(), y_hat_h.max())
    plt.scatter(y_true_h, y_hat_h, s=12, alpha=0.6)
    plt.plot([min_h, max_h], [min_h, max_h], "r--", linewidth=1)
    plt.xlabel("Real"); plt.ylabel("Predicho"); plt.title("Real vs Predicho (Humedad)")
    plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(FIGS_DIR, "scatter_humedad.png"), dpi=200)

# =========================
# Main
# =========================
def main():
    # (Opcional) fuerza CPU y reduce logs si molestan
    # import os; os.environ["CUDA_VISIBLE_DEVICES"] = "-1"; os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

    set_seeds(SEED)
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(FIGS_DIR, exist_ok=True)

    print("📦 Cargando datos…")
    X_train, X_test, y_train_temp, y_train_hum, y_test_temp, y_test_hum = cargar_datos(DATA_DIR)
    n_samples, n_steps, n_features = X_train.shape
    print(f"🔢 X_train: {X_train.shape} | X_test: {X_test.shape} | pasos={n_steps} | features={n_features}")

    print("🧠 Construyendo modelo…")
    model = construir_modelo(n_steps, n_features)
    model.summary()

    print("🚀 Entrenando (validación temporal interna)…")
    hist = entrenar(model, X_train, y_train_temp, y_train_hum)

    print("📥 Cargando escaladores de salida…")
    esc_t, esc_h = cargar_escaladores(DATA_DIR)

    print("🧪 Evaluando en TEST…")
    (yhat_t, yhat_h), (ytrue_t, ytrue_h), metrics = evaluar(model, X_test, y_test_temp, y_test_hum, esc_t, esc_h)

    print("🖼️ Guardando gráficas…")
    graficar_historial(hist)
    graficar_metricas(hist)   # << NUEVO
    graficar_predicciones(ytrue_t, ytrue_h, yhat_t, yhat_h)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_path = os.path.join(MODELS_DIR, f"modelo_multi_salida_{ts}.keras")
    model.save(final_path)
    print(f"\n💾 Modelo guardado: {final_path}")
    print("✅ Listo.")

if __name__ == "__main__":
    main()
