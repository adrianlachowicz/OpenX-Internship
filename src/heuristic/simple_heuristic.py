import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

# Load data from the file
data = pd.read_csv("../../data/covtype.data", header=None)

# Split the dataset to input data and targets
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Get the 'elevation' column
elevations_ = X.iloc[:, 0].values


def simple_heuristic(elevations: np.array):
    """
    The function gets an array of elevations and applies simple heuristic predictions on inputs.

    Arguments:
        elevations (np.array) - The array contains elevations from the dataset.

    Returns:
        outputs (list) - A list of predictions.
    """
    outputs = []

    for elev in elevations:
        if elev > 3200:
            outputs.append(1)
        elif (elev >= 2800) and (elev <= 3200):
            outputs.append(2)
        else:
            outputs.append(0)

    return outputs


# Perform a classification
y_pred = simple_heuristic(elevations_)

# Calculate and display an accuracy
print("Accuracy:", accuracy_score(y, y_pred))
