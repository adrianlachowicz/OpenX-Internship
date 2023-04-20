import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


def load_and_split_dataset(dataset_path: str, train_size: float):
    """
    The function loads datasets to the Pandas DataFrame and splits them into train and test sets.
    It also normalizes values in a dataset.

    Args:
        dataset_path (str) - A path to the dataset file.
        train_size (float) - The size of the training dataset (from 0 to 1).

    Returns:
        train_dataset (tf.data.Dataset) - A test dataset.
        test_dataset (tf.data.Dataset) - A test dataset.
    """
    test_size = round(1 - train_size, 1)

    data = pd.read_csv(dataset_path, header=None)

    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    # Normalize data
    scaler = MinMaxScaler()

    columns_to_normalize = X.columns
    X[columns_to_normalize] = scaler.fit_transform(X[columns_to_normalize])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))

    return train_dataset, test_dataset
