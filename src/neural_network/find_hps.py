import optuna
from model import build_model
from tensorflow import keras
from dataset import load_and_split_dataset
from optuna.integration import TFKerasPruningCallback

EPOCHS = 5


def objective(trial: optuna.Trial):

    # Suggest hyper-parameters
    num_hidden_layers = trial.suggest_int("num_hidden_layers", 1, 6)
    is_dropout = trial.suggest_categorical("is_dropout", [True, False])
    dropout_value = trial.suggest_categorical("dropout_value", [0.5, 0.3])
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True)

    # Load datasets
    X_train, X_test, y_train, y_test = load_and_split_dataset(
        "../../data/covtype.data", 0.7
    )

    # Build a model based on suggested parameters
    model = build_model("test", num_hidden_layers, is_dropout, dropout_value)

    # Compile the model
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                  loss="sparse_categorical_crossentropy",
                  metrics="accuracy")

    # Get 30% of the base train dataset
    train_limit = int(len(X_train) * 0.3)
    X_train = X_train[:train_limit]
    y_train = y_train[:train_limit]

    # Get 15% of the base test dataset
    test_limit = int(len(X_test) * 0.15)
    X_test = X_test[:test_limit]
    y_test = y_test[:test_limit]

    # Train model and save a history of the training
    history = model.fit(X_train, y_train,
                        epochs=EPOCHS,
                        verbose=0,
                        validation_freq=1,
                        validation_data=(X_test, y_test),
                        callbacks=[TFKerasPruningCallback(trial, "val_accuracy")])

    return history.history["val_accuracy"][-1]  # Return last validation accuracy


# Create Optuna study
study = optuna.create_study(direction="maximize",
                            storage="sqlite:///optuna_hp.sqlite3",
                            study_name="tensorflow_model")

# Run optimizing parameters
study.optimize(objective, n_trials=100, show_progress_bar=True)
