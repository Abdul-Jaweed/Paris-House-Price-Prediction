import pandas as pd
import os
from sklearn.preprocessing import RobustScaler
import pickle

BASE_DIR = "artifact"
DATASET_DIR = "data_ingestion/dataset"
TRAIN_CSV_FILE_NAME = "train.csv"
TEST_CSV_FILE_NAME = "test.csv"


def data_transformation():
    print("Performing data transformation...")
    # Load train and test DataFrames from CSV files
    train_csv_path = os.path.join(BASE_DIR, DATASET_DIR, TRAIN_CSV_FILE_NAME)
    test_csv_path = os.path.join(BASE_DIR, DATASET_DIR, TEST_CSV_FILE_NAME)
    train_df = pd.read_csv(train_csv_path)
    test_df = pd.read_csv(test_csv_path)

    # Perform the data transformation: Subtract the year (all numerical columns) from 2023
    train_df['made'] = 2023 - train_df['made']
    test_df['made'] = 2023 - test_df['made']

    # Separate features and target column in train and test sets
    X_train = train_df.drop(columns=["price"])  # Features (excluding target column)
    y_train = train_df["price"]  # Target column
    X_test = test_df.drop(columns=["price"])  # Features (excluding target column)
    y_test = test_df["price"]  # Target column

    # Apply RobustScaler to the features
    print("Performing RobustScaler")
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Save the preprocessed data as pickle files
    pickle_dir = os.path.join(BASE_DIR, "data_transformation")
    os.makedirs(pickle_dir, exist_ok=True)

    train_dir = os.path.join(pickle_dir, "train")
    os.makedirs(train_dir, exist_ok=True)

    test_dir = os.path.join(pickle_dir, "test")
    os.makedirs(test_dir, exist_ok=True)

    X_train_pkl_file = os.path.join(train_dir, "X_train.pkl")
    with open(X_train_pkl_file, "wb") as f:
        pickle.dump(X_train_scaled, f)

    y_train_pkl_file = os.path.join(train_dir, "y_train.pkl")
    with open(y_train_pkl_file, "wb") as f:
        pickle.dump(y_train, f)

    X_test_pkl_file = os.path.join(test_dir, "X_test.pkl")
    with open(X_test_pkl_file, "wb") as f:
        pickle.dump(X_test_scaled, f)

    y_test_pkl_file = os.path.join(test_dir, "y_test.pkl")
    with open(y_test_pkl_file, "wb") as f:
        pickle.dump(y_test, f)
