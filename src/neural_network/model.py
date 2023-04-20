import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from dataset import load_and_split_dataset


def build_model(name: str, hidden_layers: int, use_dropout: bool, dropout_value: float):
    """
    The function builds a neural network for classification based on the Cover Type dataset.
    The model is dynamically created using passed parameters.

    Arguments:
        name (str) - A model name.
        hidden_layers (int) - A count of hidden layers in model (from 1 to 6).
        use_dropout (bool) - Whether to use dropout layers or not.
        dropout_value (float) - A value of dropout (0.3 or 0.5).

    Returns:
        model (tf.keras.Sequential) - The model.
    """
    model = Sequential(name=name)

    input_dim = 54
    output_dim = 7

    for i in range(hidden_layers):
        hidden_dim = 2 ** (5+(hidden_layers - i-1))

        if i == 0:
            # Create input layer
            model.add(
                Dense(hidden_dim, activation="relu", input_shape=(input_dim,))
            )

            # If user wants dropout, add it
            if use_dropout:
                model.add(
                    Dropout(dropout_value)
                )

        elif i == (hidden_layers - 1):
            # Create output layer
            model.add(
                Dense(output_dim, activation="softmax")
            )
        else:
            # Create hidden layers
            model.add(
                Dense(hidden_dim, activation="relu")
            )

            # If user wants dropout, add it
            if use_dropout:
                model.add(
                    Dropout(dropout_value)
                )

    return model



X_train, X_test, y_train, y_test = load_and_split_dataset(
    "../../data/covtype.data", 0.7
)

model = build_model("test", 4, False, 0.5)
model.compile(optimizer="Adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])


print(model.summary())
model.fit(X_train, y_train)