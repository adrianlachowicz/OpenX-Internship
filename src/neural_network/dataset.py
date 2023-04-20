import pandas as pd
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
        X_train (np.array) - A train X values.
        X_test (np.array) - A train Y values.
        y_train (np.array) - An train X values.
        y_test (np.array) - An test Y values.
    """
    test_size = round(1 - train_size, 1)

    data = pd.read_csv(dataset_path, header=None)

    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values - 1

    # Normalize data
    scaler = MinMaxScaler()

    X = scaler.fit_transform(X)

    # Split data to train/test datasets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    return X_train, X_test, y_train, y_test
