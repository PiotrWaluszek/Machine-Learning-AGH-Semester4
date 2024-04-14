import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error
import matplotlib.pyplot as plt
import pickle

def generate_synthetic_data():
    np.random.seed(42)
    size = 300
    X = np.random.rand(size)*5-2.5
    w4, w3, w2, w1, w0 = 1, 2, 1, -4, 2
    y = w4*(X**4) + w3*(X**3) + w2*(X**2) + w1*X + w0 + np.random.randn(size)*8-4
    df = pd.DataFrame({'x': X, 'y': y})
    return df

def find_best_tree_depth_classification(X_train, y_train, X_test, y_test):
    best_train_depth_score = [0, 0]
    for i in range(1, 11):
        model = DecisionTreeClassifier(max_depth=i, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        current_score = f1_score(y_test, y_pred)
        if(current_score > best_train_depth_score[1]):
            best_train_depth_score = [i, current_score]
    return best_train_depth_score

def find_best_tree_depth_regression(X_train, y_train, X_test, y_test):
    best_train_depth_score = [0, float('inf')]
    for i in range(1, 11):
        regressor = DecisionTreeRegressor(max_depth=i, random_state=42)
        regressor.fit(X_train, y_train)
        y_pred = regressor.predict(X_test)
        current_score = mean_squared_error(y_test, y_pred)
        if(current_score < best_train_depth_score[1]):
            best_train_depth_score = [i, current_score]
    return best_train_depth_score

def plot_and_save_tree(model, filename):
    plt.figure(figsize=(20,10))
    plot_tree(model, filled=True)
    plt.savefig(filename)
    plt.close()

if __name__ == "__main__":
    # Generate synthetic data for regression
    df = generate_synthetic_data()

    # Load and preprocess breast cancer dataset for classification
    X, y = load_breast_cancer(return_X_y=True, as_frame=True)
    X_selected = X[['mean texture', 'mean symmetry']]
    X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=44)

    # Find the best tree depth for classification
    best_depth_classification = find_best_tree_depth_classification(X_train, y_train, X_test, y_test)
    model_classification = DecisionTreeClassifier(max_depth=best_depth_classification[0], random_state=42)
    model_classification.fit(X_train, y_train)

    # Plot and save the classification tree
    plot_and_save_tree(model_classification, 'bc.png')

    # Split the synthetic data for regression
    X_train, X_test, y_train, y_test = train_test_split(df['x'].values.reshape(-1, 1), df['y'].values, test_size=0.2, random_state=50)

    # Find the best tree depth for regression
    best_depth_regression = find_best_tree_depth_regression(X_train, y_train, X_test, y_test)
    model_regression = DecisionTreeRegressor(max_depth=best_depth_regression[0], random_state=42)
    model_regression.fit(X_train, y_train)

    # Plot and save the regression tree
    plot_and_save_tree(model_regression, 'reg.png')
