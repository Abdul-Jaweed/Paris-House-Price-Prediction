import os
import pickle
import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import math

BASE_DIR = "artifact"
MODEL_DIR = "model_trained"
TEST_PICKLE_DIR = os.path.join(BASE_DIR, "data_transformation/test")
EVALUATION_DIR = os.path.join(BASE_DIR, "model_evaluation")

def load_data_from_pickle(pickle_dir, file_name):
    file_path = os.path.join(pickle_dir, file_name)
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    return data

def load_model_from_pickle(model_dir, model_name):
    model_file = os.path.join(model_dir, f"{model_name}.pkl")
    with open(model_file, "rb") as f:
        model = pickle.load(f)
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = math.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    return mae, mse, rmse, r2

def model_evaluation():
    try:
        # Load preprocessed test data from pickle files
        X_test = load_data_from_pickle(TEST_PICKLE_DIR, "X_test.pkl")
        y_test = load_data_from_pickle(TEST_PICKLE_DIR, "y_test.pkl")

        # Load the trained models
        model_dir = os.path.join(BASE_DIR, MODEL_DIR)
        model_files = os.listdir(model_dir)

        evaluation_results = {}
        for model_file in model_files:
            model_name = os.path.splitext(model_file)[0]
            model = load_model_from_pickle(model_dir, model_name)

            # Evaluate the model
            mae, mse, rmse, r2 = evaluate_model(model, X_test, y_test)
            evaluation_results[model_name] = {
                "MAE": mae,
                "MSE": mse,
                "RMSE": rmse,
                "R2": r2
            }

            # Log the evaluation metrics using MLflow
            with mlflow.start_run(run_name=model_name):
                mlflow.log_metrics({
                    "MAE": mae,
                    "MSE": mse,
                    "RMSE": rmse,
                    "R2": r2
                })

        # Determine the best model based on the highest R2 score
        best_model = max(evaluation_results, key=lambda x: evaluation_results[x]["R2"])

        # Print the evaluation results
        print("Model Evaluation Results:")
        for model_name, metrics in evaluation_results.items():
            print(f"{model_name}: MAE={metrics['MAE']:.2f}, MSE={metrics['MSE']:.2f}, "
                  f"RMSE={metrics['RMSE']:.2f}, R2={metrics['R2']:.2f}")

        print("\nBest Model:")
        print(f"{best_model} has the highest R2={evaluation_results[best_model]['R2']:.2f}")

        # Save evaluation results as a text file
        os.makedirs(EVALUATION_DIR, exist_ok=True)
        eval_text_file = os.path.join(EVALUATION_DIR, "evaluation_results.txt")
        with open(eval_text_file, "w") as f:
            for model_name, metrics in evaluation_results.items():
                f.write(f"{model_name}: MAE={metrics['MAE']:.2f}, MSE={metrics['MSE']:.2f}, "
                        f"RMSE={metrics['RMSE']:.2f}, R2={metrics['R2']:.2f}\n")
            f.write(f"\nBest Model:\n{best_model} has the highest R2={evaluation_results[best_model]['R2']:.2f}")

        # Save evaluation results as a DataFrame file (if desired)
        eval_df = pd.DataFrame(evaluation_results).T
        eval_df_file = os.path.join(EVALUATION_DIR, "evaluation_results.pkl")
        eval_df.to_pickle(eval_df_file)

        # Log the evaluation results using MLflow
        with mlflow.start_run(run_name="Model Evaluation"):
            mlflow.log_artifact(eval_text_file)
            mlflow.log_artifact(eval_df_file)

    except Exception as e:
        print(f"An error occurred during model evaluation: {e}")

if __name__ == '__main__':
    # Set the MLflow tracking URI to the local directory
    mlflow.set_tracking_uri("file:./mlruns")
    
    # Start an MLflow run for model evaluation
    with mlflow.start_run(run_name="Model Evaluation"):
        model_evaluation()
