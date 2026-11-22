import json
import logging
import os
import argparse

import dagshub
import joblib
import mlflow
import mlflow.sklearn


# ---------------- Logging setup ----------------

logger = logging.getLogger("model_registration")
logger.setLevel(logging.DEBUG)

formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
console_handler.setFormatter(formatter)

if not logger.handlers:
    logger.addHandler(console_handler)


# ---------------- DagsHub setup ----------------

def setup_dagshub() -> None:
    """Initialize DagsHub and configure the MLflow tracking server."""
    try:
        logger.debug("Initializing DagsHub")

        # dagshub.init(
        #     repo_owner="jubito27",
        #     repo_name="mlops-mini-project",
        #     mlflow=True,
        # )

        # mlflow.set_tracking_uri(
        #     "https://dagshub.com/jubito27/mlops-mini-project.mlflow"
        # )

        # Set up DagsHub credentials for MLflow tracking
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


        logger.debug("DagsHub initialized successfully")

    except Exception:
        logger.exception("Failed to set up connection with DagsHub")
        raise


# ---------------- Model registration ----------------

def load_model_info(file_path: str) -> dict:
    """Load model information from a JSON file."""
    logger.debug("Loading model info from: %s", file_path)

    if not os.path.isfile(file_path):
        logger.error("Model info file does not exist: %s", file_path)
        raise FileNotFoundError(f"Model info file does not exist: {file_path}")

    try:
        with open(file_path, "r", encoding="utf-8") as file:
            model_info = json.load(file)

        logger.debug("Model info loaded: %s", model_info)
        return model_info

    except json.JSONDecodeError as e:
        logger.exception("Invalid JSON in model info file: %s", file_path)
        raise ValueError(f"Invalid JSON in model info file: {file_path}") from e

    except Exception:
        logger.exception("Unexpected error while loading model info")
        raise


def register_model(
    model_name: str,
    model_path: str,
    metadata_path: str,
) -> dict:
    """Log the trained pipeline and register it with a Staging alias."""
    setup_dagshub()

    mlflow.set_experiment("dvc-pipeline")

    try:
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Trained model does not exist: {model_path}")
        if not os.path.isfile(metadata_path):
            raise FileNotFoundError(f"Model metadata does not exist: {metadata_path}")

        with open(metadata_path, "r", encoding="utf-8") as file:
            metadata = json.load(file)

        pipeline = joblib.load(model_path)
        if pipeline is None:
            raise ValueError(f"Loaded model pipeline is None: {model_path}")

        with mlflow.start_run(run_name="model_registration") as run:
            mlflow.set_tags(
                {
                    "project": "gurgaon_estate",
                    "stage": "model_registry",
                    "model_type": metadata.get("model_type", "unknown"),
                }
            )
            mlflow.log_params(
                {
                    "model_name": model_name,
                    "model_path": model_path,
                    "metadata_path": metadata_path,
                }
            )
            mlflow.log_artifact(metadata_path, artifact_path="model_metadata")

            model_info = mlflow.sklearn.log_model(
                sk_model=pipeline,
                name="real_estate_prediction_pipeline",
                serialization_format="cloudpickle",
            )

            model_version = mlflow.register_model(
                model_uri=model_info.model_uri,
                name=model_name,
            )

            client = mlflow.tracking.MlflowClient()
            client.set_registered_model_alias(
                name=model_name,
                alias="Staging",
                version=model_version.version,
            )

            registration_info = {
                "model_name": model_name,
                "model_version": str(model_version.version),
                "model_uri": model_info.model_uri,
                "run_id": run.info.run_id,
                "alias": "Staging",
            }

        logger.info(
            "Model %s version %s registered with the Staging alias",
            model_name,
            model_version.version,
        )
        return registration_info

    except Exception:
        logger.exception("Error during model registration")
        raise


def main(model_dir: str, output_path: str, model_name: str) -> None:
    try:
        model_path = os.path.join(model_dir, "pipeline.pkl")
        metadata_path = os.path.join(model_dir, "model_metadata.json")
        registration_info = register_model(
            model_name=model_name,
            model_path=model_path,
            metadata_path=metadata_path,
        )

        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as file:
            json.dump(registration_info, file, indent=2)

        logger.info("Registration details saved to: %s", output_path)

    except FileNotFoundError:
        logger.exception("Required trained model artifact was not found")
        raise

    except Exception:
        logger.exception("Failed to complete the model-registration process")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Register the trained real-estate model with MLflow."
    )
    parser.add_argument(
        "--model-dir",
        default="model",
        help="Directory containing pipeline.pkl and model_metadata.json.",
    )
    parser.add_argument(
        "--output-path",
        default="reports/model_registry/registration.json",
        help="Path for the registration metadata JSON output.",
    )
    parser.add_argument(
        "--model-name",
        default="gurgaon-real-estate-pipeline",
        help="Name used for the MLflow registered model.",
    )
    args = parser.parse_args()

    main(
        model_dir=args.model_dir,
        output_path=args.output_path,
        model_name=args.model_name,
    )