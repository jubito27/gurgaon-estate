import argparse
import hashlib
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import dagshub
import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ---------------- Logging setup ----------------

LOGGER_NAME = "predict_model"


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """Configure console logging for the prediction stage."""
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(level)

    if logger.handlers:
        return logger

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)

    formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


logger = setup_logging(logging.INFO)


# ---------------- MLflow configuration ----------------

MLFLOW_EXPERIMENT_NAME = "dvc-pipeline"


def configure_mlflow() -> None:
    """Initialize DagsHub and configure the MLflow tracking server."""
    try:

        dagshub_token = os.getenv("DAGSHUB_PAT")
        if not dagshub_token:
            raise EnvironmentError("DAGSHUB_PAT environment variable is not set")

        os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
        os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

        dagshub_url = "https://dagshub.com"
        repo_owner = "jubito27"
        repo_name = "gurgaon-estate"
        # Set up MLflow tracking URI
        mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')
        mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

        logger.info("MLflow tracking configured successfully")
        logger.debug("MLflow tracking URI: %s", mlflow.get_tracking_uri())

    except Exception as error:
        logger.exception("Failed to configure MLflow tracking")
        raise RuntimeError(f"Failed to configure MLflow tracking: {error}") from error

# ---------------- Utility functions ----------------

def calculate_file_sha256(file_path: str) -> str:
    """Calculate the SHA-256 hash of a file for data lineage."""
    sha256_hash = hashlib.sha256()

    with open(file_path, "rb") as file:
        for chunk in iter(lambda: file.read(8192), b""):
            sha256_hash.update(chunk)

    return sha256_hash.hexdigest()


def validate_input_data(
    df: pd.DataFrame,
    required_columns: list[str],
    source_name: str,
) -> None:
    """Validate that the prediction input contains required columns."""
    if df.empty:
        raise ValueError(f"{source_name} is empty")

    missing_columns = [
        column for column in required_columns if column not in df.columns
    ]

    if missing_columns:
        raise KeyError(
            f"Missing required columns in {source_name}: {missing_columns}"
        )


def load_model_metadata(metadata_path: str) -> Dict[str, Any]:
    """Load model metadata created during model building."""
    if not os.path.isfile(metadata_path):
        raise FileNotFoundError(f"Model metadata file does not exist: {metadata_path}")

    with open(metadata_path, "r", encoding="utf-8") as metadata_file:
        metadata = json.load(metadata_file)

    logger.debug("Loaded model metadata: %s", metadata)
    return metadata


def load_pipeline(pipeline_path: str):
    """Load the complete fitted preprocessing and prediction pipeline."""
    if not os.path.isfile(pipeline_path):
        raise FileNotFoundError(f"Pipeline file does not exist: {pipeline_path}")

    pipeline = joblib.load(pipeline_path)

    if pipeline is None:
        raise ValueError(f"Loaded pipeline is None: {pipeline_path}")

    logger.info("Prediction pipeline loaded successfully: %s", pipeline_path)
    return pipeline


# ---------------- Prediction functions ----------------

def make_predictions(
    pipeline,
    input_df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate log-price predictions and convert them to original price."""
    logger.debug("Generating predictions for %d rows", input_df.shape[0])

    predicted_log_price = pipeline.predict(input_df)
    predicted_price = np.expm1(predicted_log_price)

    # Prevent invalid negative property prices.
    predicted_price = np.maximum(predicted_price, 0)

    logger.debug("Prediction generation completed")
    return predicted_log_price, predicted_price


def calculate_metrics(
    actual_price: np.ndarray,
    predicted_price: np.ndarray,
) -> Dict[str, float]:
    """Calculate regression metrics when actual prices are available."""
    mae = mean_absolute_error(actual_price, predicted_price)
    mse = mean_squared_error(actual_price, predicted_price)
    rmse = mse ** 0.5
    r2 = r2_score(actual_price, predicted_price)

    metrics = {
        "prediction_mae": mae,
        "prediction_mse": mse,
        "prediction_rmse": rmse,
        "prediction_r2": r2,
    }

    logger.info("Prediction metrics: %s", metrics)
    return metrics


# ---------------- MLflow tracking ----------------

def track_prediction_run(
    pipeline_path: str,
    preprocessor_path: str,
    metadata_path: str,
    input_path: str,
    predictions_path: str,
    input_df: pd.DataFrame,
    predictions_df: pd.DataFrame,
    predicted_log_price: np.ndarray,
    predicted_price: np.ndarray,
    model_metadata: Dict[str, Any],
    model_version: Optional[str],
    environment: str,
) -> str:
    """Track all prediction inputs, outputs, artifacts, and metrics in MLflow."""
    feature_df = input_df.drop(columns=["price"], errors="ignore")
    input_file_hash = calculate_file_sha256(input_path)
    pipeline_file_hash = calculate_file_sha256(pipeline_path)

    run_name = (
        f"prediction_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    )

    with mlflow.start_run(run_name=run_name) as run:
        run_id = run.info.run_id

        mlflow.set_tags(
            {
                "project": "gurgaon_estate",
                "stage": "prediction",
                "environment": environment,
                "model_type": model_metadata.get("model_type", "XGBRegressor"),
                "pipeline_type": "preprocessor+xgboost",
                "model_version": model_version or "local_pipeline",
                "input_file_sha256": input_file_hash,
                "pipeline_file_sha256": pipeline_file_hash,
            }
        )

        mlflow.log_params(
            {
                "input_path": input_path,
                "predictions_path": predictions_path,
                "pipeline_path": pipeline_path,
                "preprocessor_path": preprocessor_path,
                "metadata_path": metadata_path,
                "n_prediction_rows": input_df.shape[0],
                "n_input_features": feature_df.shape[1],
                "target_transformation": "log1p",
                "prediction_transformation": "expm1",
                "model_version": model_version or "local_pipeline",
                "environment": environment,
            }
        )

        # Log evaluation metrics only if actual prices are available.
        if "price" in input_df.columns:
            metrics = calculate_metrics(
                actual_price=input_df["price"].to_numpy(),
                predicted_price=predicted_price,
            )

            mlflow.log_metrics(metrics)

        # Log the input data used for prediction.
        mlflow.log_artifact(
            input_path,
            artifact_path="prediction_inputs",
        )

        # Log generated predictions.
        mlflow.log_artifact(
            predictions_path,
            artifact_path="prediction_outputs",
        )

        # Log model metadata for lineage and reproducibility.
        mlflow.log_artifact(
            metadata_path,
            artifact_path="model_metadata",
        )

        # Log the exact prediction source code used for this run.
        mlflow.log_artifact(
            __file__,
            artifact_path="source_code",
        )

        # Log copies of the local model artifacts for complete traceability.
        mlflow.log_artifact(
            pipeline_path,
            artifact_path="model_artifacts",
        )

        mlflow.log_artifact(
            preprocessor_path,
            artifact_path="model_artifacts",
        )

        # Log the complete pipeline as an MLflow model.
        input_example = feature_df.head(5)
        output_example = predicted_log_price[:5]

        signature = mlflow.models.infer_signature(
            model_input=input_example,
            model_output=output_example,
        )

        model_info = mlflow.sklearn.log_model(
            sk_model=joblib.load(pipeline_path),
            name="real_estate_prediction_pipeline",
            signature=signature,
            input_example=input_example,
            serialization_format="cloudpickle",
        )

        mlflow.log_param("logged_model_uri", model_info.model_uri)

        logger.info("MLflow prediction run completed successfully")
        logger.info("MLflow run ID: %s", run_id)
        logger.info("Logged model URI: %s", model_info.model_uri)

        return run_id


# ---------------- Main pipeline ----------------

def predict_model(
    input_path: str,
    output_path: str,
    output_dir: str,
    model_dir: str,
    model_version: Optional[str] = None,
    environment: str = "development",
) -> pd.DataFrame:
    """Load the saved pipeline, generate predictions, and track the run."""
    logger.info("Starting prediction pipeline")

    pipeline_path = os.path.join(model_dir, "pipeline.pkl")
    preprocessor_path = os.path.join(model_dir, "preprocessor.pkl")
    metadata_path = os.path.join(model_dir, "model_metadata.json")

    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Prediction input file does not exist: {input_path}")

    if not os.path.isfile(pipeline_path):
        raise FileNotFoundError(f"Pipeline file does not exist: {pipeline_path}")

    if not os.path.isfile(preprocessor_path):
        raise FileNotFoundError(
            f"Preprocessor file does not exist: {preprocessor_path}"
        )

    if not os.path.isfile(metadata_path):
        raise FileNotFoundError(
            f"Model metadata file does not exist: {metadata_path}"
        )

    os.makedirs(output_dir, exist_ok=True)

    input_df = pd.read_csv(input_path)

    model_metadata = load_model_metadata(metadata_path=metadata_path)

    required_columns = (
        model_metadata.get("numeric_columns", [])
        + model_metadata.get("ordinal_columns", [])
        + model_metadata.get("one_hot_columns", [])
        + model_metadata.get("target_encoding_columns", [])
    )

    validate_input_data(
        df=input_df,
        required_columns=required_columns,
        source_name="prediction input data",
    )

    pipeline = load_pipeline(pipeline_path=pipeline_path)
    feature_df = input_df.drop(columns=["price"], errors="ignore")

    predicted_log_price, predicted_price = make_predictions(
        pipeline=pipeline,
        input_df=feature_df,
    )

    predictions_df = input_df.copy()
    predictions_df["predicted_log_price"] = predicted_log_price
    predictions_df["predicted_price"] = predicted_price

    predictions_df.to_csv(output_path, index=False)

    logger.info("Predictions saved successfully: %s", output_path)
    logger.debug("Prediction output shape: %s", predictions_df.shape)

    run_id = track_prediction_run(
        pipeline_path=pipeline_path,
        preprocessor_path=preprocessor_path,
        metadata_path=metadata_path,
        input_path=input_path,
        predictions_path=output_path,
        input_df=input_df,
        predictions_df=predictions_df,
        predicted_log_price=predicted_log_price,
        predicted_price=predicted_price,
        model_metadata=model_metadata,
        model_version=model_version,
        environment=environment,
    )

    predictions_df["mlflow_run_id"] = run_id

    # Save the final output again so each prediction row contains its run ID.
    predictions_df.to_csv(output_path, index=False)

    logger.info("Prediction pipeline completed successfully")
    return predictions_df


# ---------------- CLI entry point ----------------

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate real-estate price predictions and track them in MLflow."
    )

    parser.add_argument(
        "--input-path",
        required=True,
        help="Path to the CSV file containing prediction features.",
    )

    parser.add_argument(
        "--output-path",
        required=True,
        help="Path where prediction results will be saved.",
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where prediction output files will be stored.",
    )

    parser.add_argument(
        "--model-dir",
        required=True,
        help="Directory containing pipeline.pkl, preprocessor.pkl, and model_metadata.json.",
    )

    parser.add_argument(
        "--model-version",
        default=None,
        help="Optional MLflow model version or alias used for prediction.",
    )

    parser.add_argument(
        "--environment",
        default="development",
        choices=["development", "staging", "production"],
        help="Deployment environment associated with this prediction run.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    try:
        configure_mlflow()

        predictions = predict_model(
            input_path=args.input_path,
            output_path=args.output_path,
            output_dir=args.output_dir,
            model_dir=args.model_dir,
            model_version=args.model_version,
            environment=args.environment,
        )

        print(
            f"Prediction completed successfully. "
            f"Generated {len(predictions)} predictions."
        )

    except Exception:
        logger.exception("Prediction pipeline failed")
        sys.exit(1)
