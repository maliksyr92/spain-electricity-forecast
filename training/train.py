"""
train.py
========
Trains an LSTM that forecasts the next 24 hours of Spain's national
electricity demand from the previous 7 days of features.

What it does, in order:
  1. Loads clean data (via data_prep.build_dataset).
  2. Splits by time: train = 2015-2016, test = 2017.
  3. Fits the scalers on the TRAINING data only (no data leakage).
  4. Turns the series into sliding-window sequences.
  5. Trains the LSTM and logs params/metrics to MLflow.
  6. Evaluates on 2017 with MAE and RMSE (in real megawatts).
  7. Saves the model + scalers + feature config to the models/ folder.

Run from the project root with:  python training/train.py
"""

import os
import json
import numpy as np
import joblib
import mlflow
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dropout, Dense
from tensorflow.keras.callbacks import EarlyStopping

from data_prep import build_dataset, FEATURE_COLUMNS, TARGET_COLUMN

# ---- Settings you can safely change -----------------------------------------
WINDOW = 168        # look back 7 days (168 hours)
HORIZON = 24        # predict the next 24 hours
EPOCHS = 20         # max passes over the data (early stopping may end sooner)
BATCH_SIZE = 64
LSTM_UNITS = 64
SEED = 42
# -----------------------------------------------------------------------------

np.random.seed(SEED)
tf.random.set_seed(SEED)
os.makedirs("models", exist_ok=True)


def make_sequences(features, target, window, horizon):
    """Build sliding windows: X = window hours of features, y = next horizon hours of load."""
    X, y = [], []
    for i in range(len(features) - window - horizon + 1):
        X.append(features[i:i + window])
        y.append(target[i + window:i + window + horizon])
    return np.array(X, dtype="float32"), np.array(y, dtype="float32")


def build_model(n_features):
    model = Sequential([
        Input(shape=(WINDOW, n_features)),
        LSTM(LSTM_UNITS),
        Dropout(0.2),
        Dense(HORIZON),          # 24 outputs = the next 24 hourly forecasts
    ])
    model.compile(optimizer="adam", loss="mse")
    return model


def main():
    # 1) Load clean data ------------------------------------------------------
    df = build_dataset()
    print(f"Loaded {len(df)} hourly rows ({df.index.min()} -> {df.index.max()})")

    # 2) Time-based split -----------------------------------------------------
    train_df = df[df.index.year <= 2016]
    test_df = df[df.index.year == 2017]

    # 3) Fit scalers on TRAINING data only ------------------------------------
    feature_scaler = MinMaxScaler().fit(train_df[FEATURE_COLUMNS].values)
    target_scaler = MinMaxScaler().fit(train_df[[TARGET_COLUMN]].values)

    def scale(d):
        f = feature_scaler.transform(d[FEATURE_COLUMNS].values).astype("float32")
        t = target_scaler.transform(d[[TARGET_COLUMN]].values).ravel().astype("float32")
        return f, t

    f_train, t_train = scale(train_df)
    f_test, t_test = scale(test_df)

    # 4) Build sequences ------------------------------------------------------
    X_train, y_train = make_sequences(f_train, t_train, WINDOW, HORIZON)
    X_test, y_test = make_sequences(f_test, t_test, WINDOW, HORIZON)
    n_features = X_train.shape[2]
    print(f"X_train {X_train.shape} | X_test {X_test.shape}")

    # 5) Train + log to MLflow ------------------------------------------------
    mlflow.set_experiment("spain-electricity-lstm")
    with mlflow.start_run():
        mlflow.log_params({
            "window": WINDOW, "horizon": HORIZON, "epochs": EPOCHS,
            "batch_size": BATCH_SIZE, "lstm_units": LSTM_UNITS,
            "n_features": n_features, "features": ",".join(FEATURE_COLUMNS),
        })

        model = build_model(n_features)
        early = EarlyStopping(monitor="val_loss", patience=4,
                              restore_best_weights=True)
        history = model.fit(
            X_train, y_train,
            validation_split=0.1,
            epochs=EPOCHS, batch_size=BATCH_SIZE,
            callbacks=[early], verbose=1,
        )
        # log the loss curve, epoch by epoch
        for epoch, (loss, val_loss) in enumerate(
                zip(history.history["loss"], history.history["val_loss"])):
            mlflow.log_metric("train_loss", loss, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)

        # 6) Evaluate on 2017 in real megawatts -------------------------------
        pred = model.predict(X_test)
        pred_mw = target_scaler.inverse_transform(pred)
        true_mw = target_scaler.inverse_transform(y_test)
        mae = mean_absolute_error(true_mw.ravel(), pred_mw.ravel())
        rmse = mean_squared_error(true_mw.ravel(), pred_mw.ravel()) ** 0.5
        print(f"Test MAE  = {mae:,.1f} MW")
        print(f"Test RMSE = {rmse:,.1f} MW")
        mlflow.log_metric("test_mae", mae)
        mlflow.log_metric("test_rmse", rmse)

        # 7) Save artifacts ---------------------------------------------------
        model.save("models/lstm_model.keras")
        joblib.dump(feature_scaler, "models/feature_scaler.pkl")
        joblib.dump(target_scaler, "models/target_scaler.pkl")
        with open("models/feature_config.json", "w") as fp:
            json.dump({"feature_columns": FEATURE_COLUMNS,
                       "target_column": TARGET_COLUMN,
                       "window": WINDOW, "horizon": HORIZON}, fp, indent=2)

        # also attach the artifacts to this MLflow run
        for art in ["models/lstm_model.keras", "models/feature_scaler.pkl",
                    "models/target_scaler.pkl", "models/feature_config.json"]:
            mlflow.log_artifact(art)

    print("Done. Artifacts saved in models/ and logged to MLflow.")


if __name__ == "__main__":
    main()
