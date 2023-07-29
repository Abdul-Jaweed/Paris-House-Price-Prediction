import pandas as pd
import os
import pickle
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    AdaBoostRegressor,
    GradientBoostingRegressor,
    BaggingRegressor,
)
from xgboost import XGBRegressor
from sklearn.svm import SVR

BASE_DIR = "artifact"
MODEL_DIR = "model_trained"
TRAIN_PICKLE_DIR = os.path.join(BASE_DIR, "data_transformation/train")

def load_data_from_pickle(pickle_dir, file_name):
    file_path = os.path.join(pickle_dir, file_name)
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    return data

def train_models(X_train, y_train):
    algorithms = [
        LinearRegression(),
        Ridge(alpha=1.0),
        Lasso(alpha=1.0),
        DecisionTreeRegressor(),
        RandomForestRegressor(),
        AdaBoostRegressor(),
        GradientBoostingRegressor(),
        BaggingRegressor(),
        XGBRegressor(),
        SVR(kernel='rbf'),
    ]

    trained_models = {}
    for algo in algorithms:
        model_name = algo.__class__.__name__
        print(f"Training {model_name}...")
        algo.fit(X_train, y_train)
        trained_models[model_name] = algo

    return trained_models

def model_trainer():
    try:
        # Load preprocessed data from pickle files
        X_train = load_data_from_pickle(TRAIN_PICKLE_DIR, "X_train.pkl")
        y_train = load_data_from_pickle(TRAIN_PICKLE_DIR, "y_train.pkl")

        # Train all models
        trained_models = train_models(X_train, y_train)

        # Save the trained models as pickle files in the model_trained folder
        model_dir = os.path.join(BASE_DIR, MODEL_DIR)
        os.makedirs(model_dir, exist_ok=True)

        for model_name, model in trained_models.items():
            model_file = os.path.join(model_dir, f"{model_name}.pkl")
            with open(model_file, "wb") as f:
                pickle.dump(model, f)

        print("Model training completed and pickle files saved.")

    except Exception as e:
        print(f"An error occurred during model training: {e}")

if __name__ == '__main__':
    model_trainer()
