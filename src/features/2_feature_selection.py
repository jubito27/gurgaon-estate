import os
import sys
import pathlib
import logging

import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder


# ---------------- Logging setup ----------------

def setup_logging(level: int = logging.DEBUG) -> logging.Logger:
    """Configure console logging at the requested level."""
    logger = logging.getLogger("feature_selection")
    logger.setLevel(level)

    # Avoid duplicate handlers if this module is imported/run repeatedly
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


logger = setup_logging(logging.DEBUG)


def load_data(data_path: str) -> pd.DataFrame:
    logger.debug("Loading data function started...")
    logger.debug("Loading data")

    if not os.path.isfile(data_path):
        logger.error("Input CSV file does not exist: %s", data_path)
        raise FileNotFoundError(f"Input CSV file does not exist: {data_path}")

    try:
        df = pd.read_csv(data_path)
        logger.debug("Data loaded successfully. Shape: %s", df.shape)
        return df

    except Exception:
        logger.exception("Failed to load data: %s", data_path)
        raise

def categorize_luxury(score):
    if pd.isna(score):
        return None

    if 0 <= score < 50:
        return "Low"
    elif 50 <= score < 150:
        return "Medium"
    elif 150 <= score <= 175:
        return "High"
    else:
        return None

def categorize_floor(floor):
    if pd.isna(floor):
        return None

    if 0 <= floor <= 2:
        return "Low Floor"
    elif 3 <= floor <= 10:
        return "Mid Floor"
    elif 11 <= floor <= 51:
        return "High Floor"
    else:
        return None

def feature_selection(df: pd.DataFrame) -> pd.DataFrame:
    required_columns = [
        "property_type",
        "sector",
        "price",
        "bedRoom",
        "bathroom",
        "balcony",
        "floorNum",
        "agePossession",
        "built_up_area",
        "study room",
        "servant room",
        "store room",
        "pooja room",
        "others",
        "furnishing_type",
        "luxury_score",
    ]

    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise KeyError(
            f"Missing required columns for feature selection: {missing_columns}"
        )

    train_df = df[required_columns].copy()

    train_df["luxury_category"] = train_df["luxury_score"].apply(categorize_luxury)
    train_df["floor_category"] = train_df["floorNum"].apply(categorize_floor)
    train_df.drop(
        columns=["floorNum", "luxury_score"],
        inplace=True,
    )

    data_label_encoded = train_df.copy()
    categorical_cols = data_label_encoded.select_dtypes(
        include=["object", "category"]
    ).columns

    for col in categorical_cols:
        encoder = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
        )
        data_label_encoded[col] = encoder.fit_transform(
            data_label_encoded[[col]]
        )
        logger.debug("Ordinal categories for %s: %s", col, encoder.categories_)

    X_label = data_label_encoded.drop(columns=["price"])
    y_label = data_label_encoded["price"]

    weak_features = ["pooja room", "study room", "others"]
    existing_weak_features = [
        column for column in weak_features if column in X_label.columns
    ]
    selected_df = X_label.drop(columns=existing_weak_features)
    selected_df["price"] = y_label

    logger.debug(
        "Feature selection completed. Dropped weak features: %s",
        existing_weak_features,
    )
    logger.debug("Selected dataframe shape: %s", selected_df.shape)
    logger.debug("Selected dataframe columns: %s", list(selected_df.columns))

    return selected_df


def save_data(
    df: pd.DataFrame,
    output_path: str,
    output_dir: str,
) -> None:
    logger.debug("save_data function started")

    if df is None:
        raise ValueError("Cannot save data because the DataFrame is None.")

    try:
        os.makedirs(output_dir, exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.debug("Data saved successfully at: %s", output_path)
    except Exception:
        logger.exception("Could not save DataFrame to %s", output_path)
        raise


def main(
    data_path: str,
    output_path: str,
    output_dir: str,
) -> None:
    try:
        logger.info("Main function execution started")

        df = load_data(data_path=data_path)
        logger.debug("Raw dataframe shape: %s", df.shape)
        logger.debug("Raw dataframe columns: %s", list(df.columns))

        selected_df = feature_selection(df=df)

        save_data(
            df=selected_df,
            output_path=output_path,
            output_dir=output_dir,
        )

        logger.info("Pipeline completed successfully")
    except Exception:
        logger.exception("Pipeline failed")
        raise


if __name__ == "__main__":
    current_file = pathlib.Path(__file__)
    home_dir = current_file.parent.parent.parent

    data_path = (
        home_dir
        / "data"
        / "processed"
        / "missing_value"
        / "gurgaon_missingvalue_removed.csv"
    )
    output_dir = home_dir / "data" / "processed" / "feature_selection"
    output_path = (
        output_dir
        / "gurgaon_properties_post_feature_selection.csv"
    )

    logger.info("Current file: %s", current_file)
    logger.info("Home directory: %s", home_dir)
    logger.debug("Data path: %s", data_path)
    logger.debug("Output directory: %s", output_dir)
    logger.debug("Output path: %s", output_path)

    main(
        data_path=str(data_path),
        output_path=str(output_path),
        output_dir=str(output_dir),
    )
