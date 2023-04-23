from tensorflow import keras
from keras.callbacks import ModelCheckpoint, TensorBoard
from model import build_model
from dataset import load_and_split_dataset


config = {
    "name": "model_name",
    "hidden_layers": 6,
    "use_dropout": False,
    "dropout_value": 0.5,
    "learning_rate": 0.000855,
    "epochs": 60
}

if __name__ == "__main__":
    # Load datasets
    X_train, X_test, y_train, y_test = load_and_split_dataset(
        "../../data/covtype.data", 0.7
    )

    # Build a model
    model = build_model(**config)

    # Compile the model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config["learning_rate"]),
        loss="sparse_categorical_crossentropy",
        metrics="accuracy",
    )

    # Define callbacks
    tensorboard = TensorBoard(log_dir="./logs/"+config["name"]+"/")
    model_checkpoint = ModelCheckpoint("./models/"+config["name"]+"/best_model.h5",
                                       monitor='val_accuracy',
                                       verbose=1,
                                       save_best_only=True,
                                       mode='max')

    # Train model
    history = model.fit(X_train, y_train, epochs=config["epochs"], validation_data=(X_test, y_test), callbacks=[model_checkpoint, tensorboard])

