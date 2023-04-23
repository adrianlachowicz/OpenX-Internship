import pickle
import numpy as np
import pandas as pd
import gradio as gr
from sklearn.preprocessing import MinMaxScaler
from src.neural_network.model import build_model
from sklearn.model_selection import train_test_split
from src.heuristic.simple_heuristic import simple_heuristic


MODELS_CHOICES = {
    "Heuristic": 0,
    "K-Nearest Neighbor Classifier": 1,
    "Decision Tree Classifier": 2,
    "Neural network": 3,
}

WILDERNESS_AREA_CHOICES = {
    "Rawah Wilderness Area": 0,
    "Neota Wilderness Area": 1,
    "Comanche Peak Wilderness Area": 2,
    "Cache la Poudre Wilderness Area": 3,
}

SOIL_TYPES_CHOICES = {
    "Cathedral family - Rock outcrop complex, extremely stony": 0,
    "Vanet - Ratake families complex, very stony": 1,
    "Haploborolis - Rock outcrop complex, rubbly": 2,
    "Ratake family - Rock outcrop complex, rubbly": 3,
    "Vanet family - Rock outcrop complex complex, rubbly": 4,
    "Vanet - Wetmore families - Rock outcrop complex, stony": 5,
    "Gothic family": 6,
    "Supervisor - Limber families complex": 7,
    "Troutville family, very stony": 8,
    "Bullwark - Catamount families - Rock outcrop complex, rubbly": 9,
    "Bullwark - Catamount families - Rock land complex, rubbly": 10,
    "Legault family - Rock land complex, stony": 11,
    "Catamount family - Rock land - Bullwark family complex, rubbly": 12,
    "Pachic Argiborolis - Aquolis complex": 13,
    "unspecified in the USFS Soil and ELU Survey": 14,
    "Cryaquolis - Cryoborolis complex": 15,
    "Gateview family - Cryaquolis complex": 16,
    "Rogert family, very stony": 17,
    "Typic Cryaquolis - Borohemists complex": 18,
    "Typic Cryaquepts - Typic Cryaquolls complex": 19,
    "Typic Cryaquolls - Leighcan family, till substratum complex": 20,
    "Leighcan family, till substratum, extremely bouldery": 21,
    "Leighcan family, till substratum - Typic Cryaquolls complex": 22,
    "Leighcan family, extremely stony": 23,
    "Leighcan family, warm, extremely stony": 24,
    "Granile - Catamount families complex, very stony": 25,
    "Leighcan family, warm - Rock outcrop complex, extremely stony": 26,
    "Leighcan family - Rock outcrop complex, extremely stony": 27,
    "Como - Legault families complex, extremely stony": 28,
    "Como family - Rock land - Legault family complex, extremely stony": 29,
    "Leighcan - Catamount families complex, extremely stony": 30,
    "Catamount family - Rock outcrop - Leighcan family complex, extremely stony": 31,
    "Leighcan - Catamount families - Rock outcrop complex, extremely stony": 32,
    "Cryorthents - Rock land complex, extremely stony": 33,
    "Cryumbrepts - Rock outcrop - Cryaquepts complex": 34,
    "Bross family - Rock land - Cryumbrepts complex, extremely stony": 35,
    "Rock outcrop - Cryumbrepts - Cryorthents complex, extremely stony": 36,
    "Leighcan - Moran families - Cryaquolls complex, extremely stony": 37,
    "Moran family - Cryorthents - Leighcan family complex, extremely stony": 38,
    "Moran family - Cryorthents - Rock land complex, extremely stony": 39,
}

FOREST_COVER_TYPE_CLASSES = {
    0: "Spruce/Fir",
    1: "Lodgepole Pine",
    2: "Ponderosa Pine",
    3: "Cottonwood/Willow",
    4: "Aspen",
    5: "Douglas-fir",
    6: "Krummholz",
}


def predict_heuristic(elevation: int):
    """
    The function makes predictions using the simple heuristic method.
    Args:
        elevation (int) - An elevation from the user.

    Returns:
        prediction (int) - A predicted label.
    """
    elevation = np.array([elevation])
    return simple_heuristic(elevation)


def predict_knn(input):
    """
    The function makes predictions using the K-Nearest Neighbor Classifier.

    Arguments:
        input (np.array) - An input vector.

    Returns:
        prediction (int) - A predicted label.
    """
    # Load the KNN Classifier
    with open("../../models/knn_best.sav", "rb") as f:
        knn = pickle.load(f)

    # Prepare data
    scaler = MinMaxScaler()
    data = pd.read_csv("../../data/covtype.data", header=None)

    # Split the dataset to input data and targets
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    # Split the whole dataset to train and test parts
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Fit scaler
    scaler.fit(X_train)

    # Scale input
    input = scaler.transform(input)

    # Make prediction
    prediction = knn.predict(input)[0] - 1

    return prediction


def predict_dtc(input):
    # Load the KNN Classifier
    with open("../../models/decision_tree_best.sav", "rb") as f:
        dtc = pickle.load(f)

    prediction = dtc.predict(input)[0] - 1

    return prediction


def predict_nn(input):
    """
    The function makes predictions using the Neural Network Classifier.

    Arguments:
        input (np.array) - An input vector.

    Returns:
        prediction (int) - A predicted label.
    """
    # Load the Neural Network Classifier
    config = {
        "name": "model",
        "hidden_layers": 6,
        "use_dropout": False,
        "dropout_value": 0.5,
    }

    model = build_model(**config)
    model.compile(
        loss="sparse_categorical_crossentropy",
        metrics="accuracy",
    )

    model.load_weights("../../models/nn_best.h5")

    # Prepare data
    scaler = MinMaxScaler()
    data = pd.read_csv("../../data/covtype.data", header=None)

    # Split the dataset to input data and targets
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    # Split the whole dataset to train and test parts
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Fit scaler
    scaler.fit(X_train)

    # Scale input
    input = scaler.transform(input)

    # Make prediction
    prediction = np.argmax(model.predict(input), axis=1)[0] + 1

    return prediction


def predict(
    model,
    spacer,
    elevation,
    aspect,
    slope,
    horizontal_dist_hydrology,
    vertical_dist_hydrology,
    horizontal_dist_roadways,
    hillshade_9,
    hillshade_noon,
    hillshade_3,
    horizontal_dist_firepoints,
    wilderness_area,
    soil_type,
):
    # Convert arguments to appropriate values
    model = MODELS_CHOICES[model]
    elevation = int(elevation)
    aspect = int(aspect)
    slope = int(slope)
    horizontal_dist_hydrology = int(horizontal_dist_hydrology)
    vertical_dist_hydrology = int(vertical_dist_hydrology)
    horizontal_dist_roadways = int(horizontal_dist_roadways)
    hillshade_9 = int(hillshade_9)
    hillshade_noon = int(hillshade_noon)
    hillshade_3 = int(hillshade_3)
    horizontal_dist_firepoints = int(horizontal_dist_firepoints)
    wilderness_area = WILDERNESS_AREA_CHOICES[wilderness_area]
    soil_type = SOIL_TYPES_CHOICES[soil_type]

    # Create a One-Hot Encoding
    wilderness_area_vector = [0] * 4
    soil_type_vector = [0] * 40

    wilderness_area_vector[wilderness_area] = 1
    soil_type_vector[soil_type] = 1

    # Create an input vector using the Numpy Array
    input_vector = [elevation, aspect, slope, horizontal_dist_hydrology, vertical_dist_hydrology, horizontal_dist_roadways,
                    hillshade_9, hillshade_noon, hillshade_3, horizontal_dist_firepoints,
                    ]
    input_vector.extend(wilderness_area_vector)
    input_vector.extend(soil_type_vector)

    input_vector = np.array([input_vector])

    # Select a model
    if model == 0:
        # Use heuristic model
        prediction = predict_heuristic(int(elevation))[0]
        label = FOREST_COVER_TYPE_CLASSES[prediction]
        return label
    elif model == 1:
        # Use K-Nearest Neighbor Classifier
        prediction = predict_knn(input_vector)
        label = FOREST_COVER_TYPE_CLASSES[prediction]
        return label
    elif model == 2:
        # Use Decision Tree Classifier
        prediction = predict_dtc(input_vector)
        label = FOREST_COVER_TYPE_CLASSES[prediction]
        return label
    elif model == 3:
        # Use Neural Network Classifier
        prediction = predict_nn(input_vector)
        label = FOREST_COVER_TYPE_CLASSES[prediction]
        return label

    return model


demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.components.Dropdown(list(MODELS_CHOICES.keys()), label="Choose model: "),
        gr.components.Textbox(show_label=False),
        gr.components.Textbox(label="Elevation", default=""),
        gr.components.Textbox(label="Aspect", default=""),
        gr.components.Textbox(label="Slope", default=""),
        gr.components.Textbox(label="Horizontal distance to hydrology", default=""),
        gr.components.Textbox(label="Vertical distance to hydrology", default=""),
        gr.components.Textbox(label="Horizontal distance to roadways", default=""),
        gr.components.Textbox(label="Hillshade at 9AM", default=""),
        gr.components.Textbox(label="Hillshade at noon", default=""),
        gr.components.Textbox(label="Hillshade at 3PM", default=""),
        gr.components.Textbox(label="Horizontal distance to fire points", default=""),
        gr.components.Dropdown(
            list(WILDERNESS_AREA_CHOICES.keys()), label="Wilderness area"
        ),
        gr.components.Dropdown(list(SOIL_TYPES_CHOICES.keys()), label="Soil type"),
    ],
    outputs=gr.outputs.Label(),
    title="Cover Type Classification",
)
demo.launch()
