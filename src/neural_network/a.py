import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from dataset import load_and_split_dataset


X_train, X_test, y_train, y_test = load_and_split_dataset(
    "../../data/covtype.data", 0.7
)

# Define the model architecture
model = Sequential(
    [
        Dense(256, activation="relu", input_shape=(54,)),
        Dense(128, activation="relu"),
        Dense(64, activation="relu"),
        Dense(7, activation="softmax"),
    ]
)

# Compile the model
model.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)

# Train the model
model.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test))
