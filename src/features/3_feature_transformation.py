import logging
import os
import pathlib
import sys

import numpy as np
import pandas as pd


# ---------------- Logging setup ----------------

def setup_logging(level: int = logging.DEBUG) -> logging.Logger:
    """
    Configure logging for this pipeline stage.

    Logging makes it easier to understand why a column was dropped or
    transformed when a notebook or scheduled DVC run behaves unexpectedly.
    """
    logger = logging.getLogger("feature_transformation")
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


logger = setup_logging(logging.DEBUG)


def load_data(data_path: str) -> pd.DataFrame:
    """
    Load the selected-feature CSV produced by the previous pipeline stage.

    A missing file is reported as a clear pipeline error rather than silently
    continuing with an empty or unrelated dataset.
    """
    logger.debug("Loading feature-selection data from: %s", data_path)

    if not os.path.isfile(data_path):
        logger.error("Input CSV file does not exist: %s", data_path)
        raise FileNotFoundError(f"Input CSV file does not exist: {data_path}")

    try:
        df = pd.read_csv(data_path)
        logger.info(
            "Feature-selection data loaded successfully. Shape: %s",
            df.shape,
        )
        return df
    except Exception:
        logger.exception("Failed to load feature-selection data: %s", data_path)
        raise


def normalize_age_possession(df: pd.DataFrame) -> pd.DataFrame:
    """
    Group detailed possession labels into notebook-style insight categories.

    The original notebook grouped "New Property" and "Relatively New" together
    as "new", and "Old Property" and "Moderately Old" together as "old".
    This reduces sparse categories before one-hot encoding.
    """
    if "agePossession" not in df.columns:
        return df

    age_mapping = {
        "Relatively New": "new",
        "New Property": "new",
        "Moderately Old": "old",
        "Old Property": "old",
        "Under Construction": "under construction",
    }

    df["agePossession"] = df["agePossession"].replace(age_mapping)
    logger.debug(
        "Normalized agePossession values: %s",
        sorted(df["agePossession"].dropna().astype(str).unique().tolist()),
    )
    return df


def encode_ordinal_text_categories(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert ordinal text fields into explicit numeric order.

    Property type, furnishing level, and luxury level have meaningful ordering.
    Sector and age possession are nominal, so they are one-hot encoded later.
    """
    ordinal_mappings = {
        "property_type": {"flat": 0, "house": 1},
        "luxury_category": {"Low": 0, "Medium": 1, "High": 2},
        "furnishing_type": {0: 0, 1: 1, 2: 2},
    }

    for column, mapping in ordinal_mappings.items():
        if column not in df.columns:
            continue

        if pd.api.types.is_object_dtype(df[column]) or pd.api.types.is_string_dtype(
            df[column]
        ):
            df[column] = df[column].replace(mapping)

    # If furnishing was not already numeric, convert the replacement safely.
    if "furnishing_type" in df.columns:
        df["furnishing_type"] = pd.to_numeric(
            df["furnishing_type"],
            errors="coerce",
        )

    logger.debug(
        "Completed ordinal text-category conversion for available columns."
    )
    return df


def feature_transformation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare the selected feature data for linear or tree-based models.

    This stage follows the insights notebook:
    1. Drop redundant columns identified during feature-selection analysis.
    2. Normalize possession categories.
    3. Encode remaining ordinal text fields.
    4. One-hot encode nominal fields.
    5. Log-transform price because property prices are right-skewed.
    6. Standardize features so scale-sensitive models are not biased.

    The returned CSV contains scaled feature columns and log_price.
    """
    if df is None:
        raise ValueError("Cannot transform a None DataFrame.")

    transformed_df = df.copy()

    if "price" not in transformed_df.columns:
        raise KeyError("Missing required target column: 'price'")

    # These columns were identified as redundant or weaker by the notebook.
    # Drop only columns that exist so this stage works with both raw and
    # already-encoded feature-selection outputs.
    redundant_columns = [
        "store room",
        "floor_category",
        "balcony",
        "pooja room",
        "study room",
        "others",
    ]
    columns_to_drop = [
        column
        for column in redundant_columns
        if column in transformed_df.columns
    ]

    if columns_to_drop:
        transformed_df.drop(columns=columns_to_drop, inplace=True)
        logger.info(
            "Dropped redundant/weak columns: %s",
            columns_to_drop,
        )

    transformed_df = normalize_age_possession(transformed_df)
    transformed_df = encode_ordinal_text_categories(transformed_df)

    # One-hot encode only nominal text columns.
    # drop_first=True follows the notebook and avoids a fully redundant
    # baseline dummy column for each category.
    nominal_columns = [
        column
        for column in ["sector", "agePossession"]
        if column in transformed_df.columns
        and (
            pd.api.types.is_object_dtype(transformed_df[column])
            or pd.api.types.is_string_dtype(transformed_df[column])
        )
    ]

    if nominal_columns:
        transformed_df = pd.get_dummies(
            transformed_df,
            columns=nominal_columns,
            drop_first=True,
        )
        logger.info(
            "One-hot encoded nominal columns: %s",
            nominal_columns,
        )

    # Boolean dummy columns from pd.get_dummies should be numeric before
    # StandardScaler-style processing.
    dummy_columns = transformed_df.select_dtypes(include=["bool"]).columns
    if len(dummy_columns) > 0:
        transformed_df[dummy_columns] = (
            transformed_df[dummy_columns].astype("float64")
        )

    # y = log(1 + price) matches np.log1p used in the notebook. Log
    # transformation makes positive skew closer to a symmetric distribution.
    if (transformed_df["price"] < 0).any():
        raise ValueError("Price values must be non-negative before log1p.")

    transformed_df["log_price"] = np.log1p(transformed_df["price"])
    transformed_df.drop(columns=["price"], inplace=True)
    logger.debug(
        "Created log_price with np.log1p(price). "
        "Log-price is used for model training."
    )

    # Separate the target before scaling. Standardizing the target would
    # change the interpretation of errors and is not needed here.
    feature_columns = [
        column for column in transformed_df.columns if column != "log_price"
    ]

    for column in feature_columns:
        transformed_df[column] = pd.to_numeric(
            transformed_df[column],
            errors="coerce",
        )

    missing_features = transformed_df[feature_columns].isna().sum()
    missing_features = missing_features[missing_features > 0]
    if len(missing_features) > 0:
        logger.warning(
            "Non-numeric or missing values remain in feature columns:\n%s",
            missing_features,
        )

    # Standardization has the same result as sklearn.preprocessing.StandardScaler
    # for population-standard-deviation scaling: z = (x - mean) / std(ddof=0).
    # This pure-pandas implementation keeps the pipeline lightweight.
    feature_means = transformed_df[feature_columns].mean()
    feature_stds = transformed_df[feature_columns].std(ddof=0)

    constant_columns = feature_stds[feature_stds == 0].index.tolist()
    if constant_columns:
        logger.info(
            "Zero-variance columns left unchanged: %s",
            constant_columns,
        )

    non_constant_columns = [
        column for column in feature_columns if column not in constant_columns
    ]

    if non_constant_columns:
        transformed_df[non_constant_columns] = (
            transformed_df[non_constant_columns]
            - feature_means[non_constant_columns]
        ) / feature_stds[non_constant_columns]

    # Keep constant columns numeric and at their original value. This avoids
    # division by zero while preserving their presence for a stable schema.
    transformed_df[constant_columns] = transformed_df[constant_columns].fillna(0)

    logger.info(
        "Feature transformation completed. Shape: %s",
        transformed_df.shape,
    )
    logger.debug(
        "Final transformed columns: %s",
        list(transformed_df.columns),
    )
    return transformed_df


def save_data(
    df: pd.DataFrame,
    output_path: str,
    output_dir: str,
) -> None:
    """
    Save the transformed data and create its output directory when needed.
    """
    logger.debug("save_data function started")

    if df is None:
        raise ValueError("Cannot save data because the DataFrame is None.")

    try:
        os.makedirs(output_dir, exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info("Transformed data saved successfully at: %s", output_path)
    except Exception:
        logger.exception("Could not save transformed DataFrame to %s", output_path)
        raise


def main(
    data_path: str,
    output_path: str,
    output_dir: str,
) -> None:
    """
    Execute the complete feature-transformation stage.
    """
    try:
        logger.info("Feature-transformation main function started")

        df = load_data(data_path=data_path)
        transformed_df = feature_transformation(df=df)

        save_data(
            df=transformed_df,
            output_path=output_path,
            output_dir=output_dir,
        )

        logger.info("Feature-transformation pipeline completed successfully")
    except Exception:
        logger.exception("Feature-transformation pipeline failed")
        raise


if __name__ == "__main__":
    current_file = pathlib.Path(__file__)
    home_dir = current_file.parent.parent.parent

    primary_input = (
        home_dir
        / "data"
        / "processed"
        / "feature_selection"
        / "gurgaon_properties_post_feature_selection.csv"
    )

    # The fallback keeps this stage usable while the feature-selection stage
    # is being installed or regenerated in environments without scikit-learn.
    # DVC should use the primary input once the 8-feature-selection stage exists.
    legacy_input = (
        home_dir
        / "data"
        / "processed"
        / "feature_transformation"
        / "gurgaon_properties_post_feature_selection.csv"
    )

    data_path = primary_input if primary_input.is_file() else legacy_input
    output_dir = home_dir / "data" / "processed" / "feature_transformation"
    output_path = output_dir / "gurgaon_properties_transformed.csv"

    if data_path == legacy_input:
        logger.warning(
            "Primary feature-selection CSV was not found. "
            "Temporarily using notebook-era fallback: %s",
            legacy_input,
        )

    logger.info("Current file: %s", current_file)
    logger.info("Home directory: %s", home_dir)
    logger.debug("Input data path: %s", data_path)
    logger.debug("Output directory: %s", output_dir)
    logger.debug("Output path: %s", output_path)

    main(
        data_path=str(data_path),
        output_path=str(output_path),
        output_dir=str(output_dir),
    )
