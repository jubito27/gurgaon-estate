import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import pandas as pd
import yaml
from dvclive import Live

from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from category_encoders import TargetEncoder
from xgboost import XGBRegressor


# ---------------- Logging setup ----------------

LOGGER_NAME = "model_building"


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """Configure console logging for the model-building stage."""
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


# ---------------- Configuration ----------------

NUMERIC_COLUMNS = [
    "bedRoom",
    "bathroom",
    "built_up_area",
    "servant room",
    "store room",
]

ORDINAL_COLUMNS = [
    "property_type",
    "balcony",
    "furnishing_type",
    "luxury_category",
    "floor_category",
]

ONE_HOT_COLUMNS = ["agePossession"]

TARGET_ENCODING_COLUMNS = ["sector"]


# ---------------- Data functions ----------------

def validate_dataframe(
    df: pd.DataFrame,
    target_column: str,
    source_name: str,
) -> None:
    """Validate that the training DataFrame is usable for regression."""
    if df.empty:
        raise ValueError(f"{source_name} is empty")

    if target_column not in df.columns:
        raise KeyError(
            f"Target column '{target_column}' is missing from {source_name}"
        )

    required_feature_columns = (
        NUMERIC_COLUMNS
        + ORDINAL_COLUMNS
        + ONE_HOT_COLUMNS
        + TARGET_ENCODING_COLUMNS
    )

    missing_columns = [
        column for column in required_feature_columns if column not in df.columns
    ]

    if missing_columns:
        raise KeyError(
            f"Missing required feature columns in {source_name}: {missing_columns}"
        )

    if df[target_column].isna().any():
        raise ValueError(f"Target column '{target_column}' contains missing values")

    if not pd.api.types.is_numeric_dtype(df[target_column]):
        raise TypeError(
            f"Target column '{target_column}' must be numeric. "
            f"Found dtype: {df[target_column].dtype}"
        )


def load_training_data(
    data_path: str,
    target_column: str,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Load one complete training dataframe and split it into features and target."""
    logger.debug("Starting training data loading")
    logger.debug("Data path: %s", data_path)

    try:
        if not os.path.isfile(data_path):
            logger.error("Input file does not exist: %s", data_path)
            raise FileNotFoundError(f"Input file does not exist: {data_path}")

        df = pd.read_csv(data_path)
        logger.debug("Training dataframe loaded. Shape: %s", df.shape)

        validate_dataframe(
            df=df,
            target_column=target_column,
            source_name="training dataframe",
        )

        y_train = df[target_column].copy()
        x_train = df.drop(columns=[target_column])

        logger.info("Training data loaded and validated successfully")
        return x_train, y_train

    except Exception:
        logger.exception("Failed to load or validate training data")
        raise


def create_train_validation_split(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    validation_size: float,
    random_state: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Create a validation split for model evaluation."""
    if not 0 < validation_size < 1:
        raise ValueError(
            f"validation_size must be between 0 and 1. Received: {validation_size}"
        )

    x_fit, x_validation, y_fit, y_validation = train_test_split(
        x_train,
        y_train,
        test_size=validation_size,
        random_state=random_state,
    )

    logger.debug(
        "Created train/validation split. Train: %s, Validation: %s",
        x_fit.shape,
        x_validation.shape,
    )

    return x_fit, x_validation, y_fit, y_validation


# ---------------- Model functions ----------------

def build_preprocessor() -> ColumnTransformer:
    """Build the preprocessing transformer for raw model features."""
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                StandardScaler(),
                NUMERIC_COLUMNS,
            ),
            (
                "ordinal",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
                ORDINAL_COLUMNS,
            ),
            (
                "age_one_hot",
                OneHotEncoder(
                    drop="first",
                    sparse_output=False,
                    handle_unknown="ignore",
                ),
                ONE_HOT_COLUMNS,
            ),
            (
                "sector_target_encoding",
                TargetEncoder(),
                TARGET_ENCODING_COLUMNS,
            ),
        ],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )

    return preprocessor


def build_pipeline(model_params: Dict[str, Any]) -> Pipeline:
    """Build an unfitted preprocessing and XGBoost regression pipeline."""
    preprocessor = build_preprocessor()

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", XGBRegressor(**model_params)),
        ]
    )

    return pipeline


def evaluate_model(
    pipeline: Pipeline,
    x_validation: pd.DataFrame,
    y_validation: pd.Series,
) -> Dict[str, float]:
    """Calculate regression metrics on the validation set."""
    predictions = pipeline.predict(x_validation)

    mae = mean_absolute_error(y_validation, predictions)
    mse = mean_squared_error(y_validation, predictions)
    rmse = mse ** 0.5
    r2 = r2_score(y_validation, predictions)

    metrics = {
        "validation_mae": mae,
        "validation_mse": mse,
        "validation_rmse": rmse,
        "validation_r2": r2,
    }

    logger.info("Validation metrics: %s", metrics)

    return metrics


def train_model(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    model_params: Dict[str, Any],
    validation_size: float,
    random_state: int,
) -> Tuple[Pipeline, Dict[str, float]]:
    """Train, validate, and refit the final XGBoost regression pipeline."""
    logger.info("Starting XGBoost regression model training")

    try:
        x_fit, x_validation, y_fit, y_validation = create_train_validation_split(
            x_train=x_train,
            y_train=y_train,
            validation_size=validation_size,
            random_state=random_state,
        )

        validation_pipeline = build_pipeline(model_params=model_params)

        logger.debug("Fitting validation pipeline")
        validation_pipeline.fit(x_fit, y_fit)

        validation_metrics = evaluate_model(
            pipeline=validation_pipeline,
            x_validation=x_validation,
            y_validation=y_validation,
        )

        # Refit on the complete training dataset after validation.
        final_pipeline = build_pipeline(model_params=model_params)

        logger.debug("Refitting final pipeline on the full training dataset")
        final_pipeline.fit(x_train, y_train)

        logger.info("XGBoost regression model training completed successfully")

        return final_pipeline, validation_metrics

    except Exception:
        logger.exception("Failed during model training")
        raise


# ---------------- Artifact functions ----------------

def save_artifacts(
    pipeline: Pipeline,
    validation_metrics: Dict[str, float],
    model_params: Dict[str, Any],
    target_column: str,
    output_dir: str,
) -> Dict[str, str]:
    """Save the fitted pipeline, preprocessor, and model metadata."""
    logger.debug("Starting artifact saving process")
    logger.debug("Model output directory: %s", output_dir)

    if pipeline is None:
        raise ValueError("Cannot save artifacts because the pipeline is None")

    try:
        os.makedirs(output_dir, exist_ok=True)
        logger.debug("Ensured model output directory exists: %s", output_dir)

        pipeline_path = os.path.join(output_dir, "pipeline.pkl")
        preprocessor_path = os.path.join(output_dir, "preprocessor.pkl")
        metadata_path = os.path.join(output_dir, "model_metadata.json")

        joblib.dump(pipeline, pipeline_path)
        logger.info("Fitted pipeline saved successfully: %s", pipeline_path)

        fitted_preprocessor = pipeline.named_steps["preprocessor"]
        joblib.dump(fitted_preprocessor, preprocessor_path)
        logger.info("Fitted preprocessor saved successfully: %s", preprocessor_path)

        metadata = {
            "target_column": target_column,
            "model_type": "XGBRegressor",
            "pipeline_steps": [step[0] for step in pipeline.steps],
            "numeric_columns": NUMERIC_COLUMNS,
            "ordinal_columns": ORDINAL_COLUMNS,
            "one_hot_columns": ONE_HOT_COLUMNS,
            "target_encoding_columns": TARGET_ENCODING_COLUMNS,
            "model_params": model_params,
            "validation_metrics": validation_metrics,
            "artifacts": {
                "pipeline": pipeline_path,
                "preprocessor": preprocessor_path,
            },
        }

        with open(metadata_path, "w", encoding="utf-8") as metadata_file:
            json.dump(metadata, metadata_file, indent=4, default=str)

        logger.info("Model metadata saved successfully: %s", metadata_path)

        artifact_paths = {
            "pipeline": pipeline_path,
            "preprocessor": preprocessor_path,
            "metadata": metadata_path,
        }

        logger.info("All model artifacts saved successfully")
        return artifact_paths

    except Exception:
        logger.exception("Failed to save model artifacts")
        raise


# ---------------- Pipeline ----------------

def main(
    data_path: str,
    params_path: str,
    output_dir: str,
) -> None:
    """Run the complete model-building pipeline."""
    logger.info("Starting model-building pipeline")

    try:
        with open(params_path, "r", encoding="utf-8") as params_file:
            params = yaml.safe_load(params_file)

        logger.debug("Loaded YAML configuration: %s", params)

        model_building_params = params["model_building"]

        target_column = model_building_params["target_column"]
        validation_size = model_building_params["validation_size"]
        random_state = model_building_params["random_state"]
        model_params = model_building_params["xgboost"]

        x_train, y_train = load_training_data(
            data_path=data_path,
            target_column=target_column,
        )

        with Live(save_dvc_exp=True) as live:
            live.log_param("target_column", target_column)
            live.log_param("validation_size", validation_size)
            live.log_param("random_state", random_state)
            live.log_param("n_training_rows", x_train.shape[0])
            live.log_param("n_features", x_train.shape[1])

            for parameter_name, parameter_value in model_params.items():
                live.log_param(f"xgboost_{parameter_name}", parameter_value)

            pipeline, validation_metrics = train_model(
                x_train=x_train,
                y_train=y_train,
                model_params=model_params,
                validation_size=validation_size,
                random_state=random_state,
            )

            for metric_name, metric_value in validation_metrics.items():
                live.log_metric(metric_name, metric_value)

            artifact_paths = save_artifacts(
                pipeline=pipeline,
                validation_metrics=validation_metrics,
                model_params=model_params,
                target_column=target_column,
                output_dir=output_dir,
            )

            live.log_artifact(artifact_paths["pipeline"], name="pipeline")
            live.log_artifact(artifact_paths["preprocessor"], name="preprocessor")
            live.log_artifact(artifact_paths["metadata"], name="model_metadata")

        logger.info("Model-building pipeline completed successfully")

    except FileNotFoundError:
        logger.exception("Required file or directory was not found")
        raise

    except KeyError as error:
        logger.exception(
            "Missing required configuration key in params.yaml: %s", error
        )
        raise

    except Exception:
        logger.exception("Unexpected error occurred during model building")
        raise


if __name__ == "__main__":
    current_file = Path(__file__)
    home_dir = current_file.parent.parent.parent

    model_dir = home_dir / "model"

    data_path = (
        home_dir
        / "data"
        / "processed"
        / "feature_selection"
        / "gurgaon_properties_post_feature_selection_v2.csv"
    )

    params_path = home_dir / "params.yaml"

    logger.debug("Project home directory: %s", home_dir)
    logger.debug("YAML configuration path: %s", params_path)
    logger.debug("Training dataframe path: %s", data_path)
    logger.debug("Model output directory: %s", model_dir)

    try:
        main(
            data_path=str(data_path),
            params_path=str(params_path),
            output_dir=str(model_dir),
        )

        print("Model training completed successfully")

    except Exception:
        logger.exception("Model-building pipeline failed")
        sys.exit(1)