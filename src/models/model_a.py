# src/models/model_a.py
import pandas as pd
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from src.clean_dataset import clean_dataset

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam


def build_model(input_dim: int):
    model = Sequential([
        Dense(32, activation="relu", input_shape=(input_dim,)),
        Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model


def train(epochs: int):
    print(f"Training Model A (SNN) for {epochs} epochs")

    try:
        df = clean_dataset("datasets/heart-disease-data/heart_disease_uci.csv")
    except FileNotFoundError:
        print("Error: File not found. Please check the file path.")
        return None, None

    X = df.drop("num", axis=1)
    y = df["num"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = build_model(X_train.shape[1])

    history = model.fit(
        X_train,
        y_train,
        validation_split=0.2,
        epochs=epochs,
        batch_size=32,
        verbose=1
    )

    y_pred_prob = model.predict(X_test)
    y_pred = (y_pred_prob > 0.5).astype(int)

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    return model, history


if __name__ == "__main__":
    train(epochs=10)
