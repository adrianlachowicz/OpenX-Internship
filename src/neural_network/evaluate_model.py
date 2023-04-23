import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow import keras
from keras.models import load_model
from dataset import load_and_split_dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split


# Load data for DTC and KNN without normalizing
data = pd.read_csv("../../data/covtype.data", header=None)

X = data.iloc[:, :-1]
y = data.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)


# Load data for neural network with normalization
X_train_n, X_test_n, y_train_n, y_test_n = load_and_split_dataset(
    "../../data/covtype.data", 0.7
)

# Load the Decision Tree Classifier
with open("../../models/decision_tree_best.sav", "rb") as f:
    dtc = pickle.load(f)

# Load the KNN Classifier
with open("../../models/knn_best.sav", "rb") as f:
    knn = pickle.load(f)

# Load the neural network
nn_model = load_model('../../models/nn_best.h5', compile=False)

nn_model.compile(
    optimizer=keras.optimizers.Adam(),
    loss="sparse_categorical_crossentropy",
    metrics="accuracy",
)


# Make predictions
y_pred1 = dtc.predict(X_test)
y_pred2 = knn.predict(X_test)
y_pred3 = np.argmax(np.round(nn_model.predict(X_test_n)), axis=1)


# Calculate accuracies
acc1 = accuracy_score(y_test, y_pred1)
acc2 = accuracy_score(y_test, y_pred2)
acc3 = accuracy_score(y_test_n, y_pred3)


# Calculate precisions
precision1 = precision_score(y_test, y_pred1, average="weighted")
precision2 = precision_score(y_test, y_pred2, average="weighted")
precision3 = precision_score(y_test_n, y_pred3, average="weighted")


# Calculate recalls
recall1 = recall_score(y_test, y_pred1, average="weighted")
recall2 = recall_score(y_test, y_pred2, average="weighted")
recall3 = recall_score(y_test_n, y_pred3, average="weighted")


# Calculate F1-scores
f1_1 = f1_score(y_test, y_pred1, average="weighted")
f1_2 = f1_score(y_test, y_pred2, average="weighted")
f1_3 = f1_score(y_test_n, y_pred3, average="weighted")


# Calculate confusion matrices
cm_1 = confusion_matrix(y_test, y_pred1)
cm_2 = confusion_matrix(y_test, y_pred2)
cm_3 = confusion_matrix(y_test_n, y_pred3)


# Create a confusion map for Decision Tree Classifier
fig, ax = plt.subplots(figsize=(8, 8))
ax = sns.heatmap(cm_1, annot=True, cmap='Blues', fmt='g', cbar=False)

ax.set_xlabel('Predicted labels')
ax.set_ylabel('True labels')
ax.set_title(f'Decision Tree - Confusion Matrix')

plt.savefig(f'confusion_matrix_dtc.png')


# Create a confusion map for KNN Classifier
fig, ax = plt.subplots(figsize=(8, 8))
ax = sns.heatmap(cm_2, annot=True, cmap='Blues', fmt='g', cbar=False)

ax.set_xlabel('Predicted labels')
ax.set_ylabel('True labels')
ax.set_title(f'KNN - Confusion Matrix')

plt.savefig(f'confusion_matrix_knn.png')


# Create a confusion map for neural network
fig, ax = plt.subplots(figsize=(8, 8))
ax = sns.heatmap(cm_3, annot=True, cmap='Blues', fmt='g', cbar=False)

ax.set_xlabel('Predicted labels')
ax.set_ylabel('True labels')
ax.set_title(f'Neural Network - Confusion Matrix')

plt.savefig(f'confusion_matrix_nn.png')


# Show results
print("Decision Tree classifier:")
print("\t Accuracy: {}".format(acc1))
print("\t Precision: {}".format(precision1))
print("\t Recall: {}".format(recall1))
print("\t F1-score: {}".format(f1_1))

print("KNN Classifier:")
print("\t Accuracy: {}".format(acc2))
print("\t Precision: {}".format(precision2))
print("\t Recall: {}".format(recall2))
print("\t F1-score: {}".format(f1_2))

print("Neural network:")
print("\t Accuracy: {}".format(acc3))
print("\t Precision: {}".format(precision3))
print("\t Recall: {}".format(recall3))
print("\t F1-score: {}".format(f1_3))
