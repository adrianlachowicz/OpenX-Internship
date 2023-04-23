import pickle
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from argparse import ArgumentParser
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import precision_score, recall_score, accuracy_score


def parse_args():
    """
    The function shows a description of the script if the user runs it with the '-h' flag.

    Returns:
        args - Arguments.
    """

    parser = ArgumentParser(
        description="The script trains a 'KNN' classifier. "
        "Also, it can predict a label based on new data (can be invoked from another script)."
    )
    parser.add_argument(
        "--model-filename",
        type=str,
        required=True,
        help="A model name to save after training.",
    )
    parser.add_argument(
        "--data-path", type=str, required=True, help="A path to the data file."
    )
    args = parser.parse_args()

    return args


def train_model(data_path: str, model_filename: str):
    """
    The function trains a specific model (in this case KNeighborsClassifier), calculates accuracy and saves it.

    Arguments:
        data_path (str) - A path to the data file.
        model_filename (str) - A model name to save after training.

    Returns:
        accuracy (float) - The accuracy.
        precision (float) - The precision.
        recall (float) - The recall.
    """

    scaler = MinMaxScaler()

    # Load data from the file
    data = pd.read_csv(data_path, header=None)

    # Split the dataset to input data and targets
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    # Split the whole dataset to train and test parts
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    classifier = KNeighborsClassifier(n_neighbors=5)

    # Perform training
    classifier.fit(X_train, y_train)

    # Create predictions for test dataset
    y_pred = classifier.predict(X_test)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="macro")
    recall = recall_score(y_test, y_pred, average="macro")

    # Save model
    model_filename = model_filename + ".sav"
    pickle.dump(classifier, open("./models/" + model_filename, "wb"))

    return accuracy, precision, recall


if __name__ == "__main__":
    args = parse_args()
    data_path = args.data_path
    model_filename = args.model_filename

    accuracy, precision, recall = train_model(data_path, model_filename)

    accuracy = round(accuracy * 100, 2)
    precision = round(precision * 100, 2)
    recall = round(recall * 100, 2)

    print("\n")
    print("-------------------------- Training summary --------------------------")
    print("Accuracy: {}%".format(accuracy))
    print("Precision: {}".format(precision))
    print("Recall: {}".format(accuracy))
    print("----------------------------------------------------------------------")
