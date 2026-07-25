"""
Project: Iris Flower Species Classification using Machine Learning
====================================================================
This project builds a machine learning model that classifies the
species of an Iris flower based on 4 measurements: sepal length,
sepal width, petal length, and petal width.

Algorithm used: Decision Tree Classifier
"""

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import pandas as pd


def load_and_explore_data():
    """Load the Iris dataset and display an overview of it."""
    iris = load_iris()

    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['species'] = iris.target

    print("=== Data Overview ===")
    print(df.head())
    print(f"\nNumber of samples: {df.shape[0]}")
    print(f"Number of features: {df.shape[1] - 1}")
    print(f"Flower species: {list(iris.target_names)}\n")

    return df, iris.target_names


def split_data(df):
    """Split the data into training and testing sets."""
    X = df.drop('species', axis=1)
    y = df['species']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("=== Data Split ===")
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}\n")

    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train):
    """Train a Decision Tree model on the training data."""
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)

    print("=== Training ===")
    print("Model trained successfully ✅\n")

    return model


def evaluate_model(model, X_test, y_test):
    """Evaluate the model's performance on the test data."""
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    print("=== Model Evaluation ===")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Model predictions: {predictions[:10]}")
    print(f"Actual labels:     {y_test.values[:10]}\n")

    return accuracy


def predict_new_sample(model, target_names, sample):
    """Predict the species of a new flower based on its measurements."""
    prediction = model.predict([sample])
    predicted_species = target_names[prediction[0]]

    print("=== Predicting a New Sample ===")
    print(f"Input measurements: {sample}")
    print(f"Predicted species: {predicted_species}\n")

    return predicted_species


def main():
    # 1. Load and explore the data
    df, target_names = load_and_explore_data()

    # 2. Split the data
    X_train, X_test, y_train, y_test = split_data(df)

    # 3. Train the model
    model = train_model(X_train, y_train)

    # 4. Evaluate the model
    evaluate_model(model, X_test, y_test)

    # 5. Try it on a new sample
    new_sample = [5.1, 3.5, 1.4, 0.2]
    predict_new_sample(model, target_names, new_sample)


if __name__ == "__main__":
    main()
