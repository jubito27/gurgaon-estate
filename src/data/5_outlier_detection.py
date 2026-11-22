import os
import sys
import pathlib
import logging

import pandas as pd
import numpy as np


# ---------------- Logging setup ----------------

def setup_logging(level: int = logging.DEBUG) -> logging.Logger:
    """Configure console logging at the requested level."""
    logger = logging.getLogger("data_preprocessing_houses")
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


def calculate_iqr_bounds(series: pd.Series) -> tuple[float, float]:
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    return lower_bound, upper_bound


def outlier_detection(df: pd.DataFrame) -> pd.DataFrame:
    logger.debug("Outlier detection function started...")

    required_columns = [
        "price",
        "price_per_sqft",
        "area",
        "bedRoom",
        "carpet_area",
    ]

    missing_columns = [
        column for column in required_columns if column not in df.columns
    ]

    if missing_columns:
        raise KeyError(
            f"Missing required columns for outlier detection: {missing_columns}"
        )

    df = df.copy()

    # ---------------- Price outliers ----------------

    price_lower, price_upper = calculate_iqr_bounds(df["price"])

    price_outliers = df[
        (df["price"] < price_lower) | (df["price"] > price_upper)
    ]

    logger.debug(
        "Price IQR bounds: lower=%s, upper=%s",
        price_lower,
        price_upper,
    )
    logger.debug("Number of price outliers: %s", price_outliers.shape[0])
    logger.debug("Price outlier statistics:\n%s", price_outliers["price"].describe())

    # The notebook treats these as a mix of genuine luxury properties and
    # data errors. It does not remove them solely based on the price IQR.
    # The actual removal happens later using explicit index labels.

    # ---------------- Price-per-sqft outliers ----------------

    ppsf_lower, ppsf_upper = calculate_iqr_bounds(df["price_per_sqft"])

    ppsf_outliers_mask = (
        (df["price_per_sqft"] < ppsf_lower)
        | (df["price_per_sqft"] > ppsf_upper)
    )

    ppsf_outliers = df.loc[ppsf_outliers_mask].copy()

    logger.debug(
        "Price-per-sqft IQR bounds: lower=%s, upper=%s",
        ppsf_lower,
        ppsf_upper,
    )
    logger.debug(
        "Number of price-per-sqft outliers: %s",
        ppsf_outliers.shape[0],
    )
    logger.debug(
        "Price-per-sqft outlier statistics:\n%s",
        ppsf_outliers["price_per_sqft"].describe(),
    )

    # For likely unit-conversion errors, correct small areas from sq.m. to sq.ft.
    small_area_mask = ppsf_outliers["area"] < 1000

    if small_area_mask.any():
        ppsf_outliers.loc[small_area_mask, "area"] = (
            ppsf_outliers.loc[small_area_mask, "area"] * 9
        )

        ppsf_outliers["price_per_sqft"] = (
            ppsf_outliers["price"] * 10000000
        ) / ppsf_outliers["area"]

        df.update(ppsf_outliers[["area", "price_per_sqft"]])

        logger.debug(
            "Corrected area and recomputed price_per_sqft for %d rows",
            small_area_mask.sum(),
        )

    # ---------------- Remove impossible values ----------------

    before_shape = df.shape
    df = df[df["price_per_sqft"] <= 50000]
    logger.debug(
        "Removed %d rows with price_per_sqft greater than 50000",
        before_shape[0] - df.shape[0],
    )

    before_shape = df.shape
    df = df[df["area"] < 100000]
    logger.debug(
        "Removed %d rows with area greater than or equal to 100000",
        before_shape[0] - df.shape[0],
    )

    # ---------------- Remove manually identified data errors ----------------

    # These index labels come from the notebook analysis.
    # Use errors="ignore" so the pipeline does not fail if an index
    # is absent after earlier filtering.
    manual_drop_indices = [
        818,
        1796,
        1123,
        2,
        2356,
        115,
        3649,
        2503,
        1471,
    ]

    existing_manual_drop_indices = [
        index for index in manual_drop_indices if index in df.index
    ]

    if existing_manual_drop_indices:
        before_shape = df.shape
        df.drop(index=existing_manual_drop_indices, inplace=True)

        logger.debug(
            "Removed %d manually identified data-error rows",
            before_shape[0] - df.shape[0],
        )

    # ---------------- Bedroom outliers ----------------

    before_shape = df.shape
    df = df[df["bedRoom"] <= 10]
    logger.debug(
        "Removed %d rows with more than 10 bedrooms",
        before_shape[0] - df.shape[0],
    )

    # ---------------- Correct known area errors ----------------

    # These corrections were identified manually in the notebook.
    area_corrections = {
        48: 1159,
        300: 7250,
        2666: 5800,
        1358: 2660,
        3195: 2850,
        2131: 1812,
        3088: 2160,
        3444: 1175,
    }

    for index, corrected_area in area_corrections.items():
        if index in df.index:
            df.loc[index, "area"] = corrected_area

    if area_corrections:
        logger.debug(
            "Applied %d manual area corrections",
            len(area_corrections),
        )

    # ---------------- Correct known carpet-area errors ----------------

    # These values were also corrected manually in the notebook.
    carpet_area_corrections = {
        2131: 1812,
        2342: 1100,
    }

    for index, corrected_carpet_area in carpet_area_corrections.items():
        if index in df.index:
            df.loc[index, "carpet_area"] = corrected_carpet_area

    if carpet_area_corrections:
        logger.debug(
            "Applied %d manual carpet-area corrections",
            len(carpet_area_corrections),
        )

    # ---------------- Area-per-bedroom outlier removal ----------------

    valid_bedroom_mask = df["bedRoom"] > 0

    reference_df = df[
        (df["price_per_sqft"] <= 20000) & valid_bedroom_mask
    ]

    if reference_df.empty:
        logger.warning(
            "No reference rows available to calculate the area-per-bedroom threshold"
        )
        area_per_bedroom_threshold = 0
    else:
        area_per_bedroom_threshold = (
            reference_df["area"] / reference_df["bedRoom"]
        ).quantile(0.02)

    logger.debug(
        "Area-per-bedroom 2nd-percentile threshold: %s",
        area_per_bedroom_threshold,
    )

    before_shape = df.shape

    df = df[
        valid_bedroom_mask
        & ((df["area"] / df["bedRoom"]) >= area_per_bedroom_threshold)
    ]

    logger.debug(
        "Removed %d rows below the area-per-bedroom threshold",
        before_shape[0] - df.shape[0],
    )

    logger.debug("Outlier detection completed. Final shape: %s", df.shape)

    return df


def save_data(df: pd.DataFrame, output_path: str, output_dir: str) -> None:
    logger.debug("save_data function started")

    if df is None:
        raise ValueError("Cannot save data because the DataFrame is None.")

    try:
        os.makedirs(output_dir, exist_ok=True)
        logger.debug("Output directory ready: %s", output_dir)

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

        df = outlier_detection(df=df)

        logger.debug("Outlier detection completed")

        save_data(
            df=df,
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

    logger.info("Current file: %s", current_file)
    logger.info("Home directory: %s", home_dir)

    data_path = home_dir / "data" / "interim" / "v1" / "gurgaon_v1.csv"
    output_dir = home_dir / "data" / "processed" / "outlier_removed"
    output_path = output_dir / "gurgaon_outlier_removed.csv"

    logger.debug("Data path: %s", data_path)
    logger.debug("Output directory: %s", output_dir)
    logger.debug("Output path: %s", output_path)

    main(
        data_path=str(data_path),
        output_path=str(output_path),
        output_dir=str(output_dir),
    )