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


def get_mode_age_possession(
    df: pd.DataFrame,
    sector: object,
    property_type: object,
    group_by_sector: bool,
    group_by_property_type: bool,
) -> object:
    """
    Return the mode agePossession for the requested group.

    If no mode is available, return NaN.
    """
    filtered_df = df

    if group_by_sector:
        filtered_df = filtered_df[filtered_df["sector"] == sector]

    if group_by_property_type:
        filtered_df = filtered_df[filtered_df["property_type"] == property_type]

    mode_values = filtered_df["agePossession"].mode(dropna=True)

    if mode_values.empty:
        return np.nan

    return mode_values.iloc[0]


def impute_age_possession_by_sector_and_property_type(row, df: pd.DataFrame):
    if pd.notna(row["agePossession"]) and row["agePossession"] != "Undefined":
        return row["agePossession"]

    return get_mode_age_possession(
        df=df,
        sector=row["sector"],
        property_type=row["property_type"],
        group_by_sector=True,
        group_by_property_type=True,
    )


def impute_age_possession_by_sector(row, df: pd.DataFrame):
    if pd.notna(row["agePossession"]) and row["agePossession"] != "Undefined":
        return row["agePossession"]

    return get_mode_age_possession(
        df=df,
        sector=row["sector"],
        property_type=row["property_type"],
        group_by_sector=True,
        group_by_property_type=False,
    )


def impute_age_possession_by_property_type(row, df: pd.DataFrame):
    if pd.notna(row["agePossession"]) and row["agePossession"] != "Undefined":
        return row["agePossession"]

    return get_mode_age_possession(
        df=df,
        sector=row["sector"],
        property_type=row["property_type"],
        group_by_sector=False,
        group_by_property_type=True,
    )


def impute_built_up_area(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute missing built_up_area using available super_built_up_area
    and carpet_area values.
    """
    all_present_mask = (
        df["super_built_up_area"].notna()
        & df["built_up_area"].notna()
        & df["carpet_area"].notna()
    )

    all_present_df = df.loc[all_present_mask]

    if all_present_df.empty:
        logger.warning(
            "No rows contain all three area fields; "
            "area-ratio-based imputation cannot be calculated"
        )
        return df

    super_to_built_up_ratio = (
        all_present_df["super_built_up_area"]
        / all_present_df["built_up_area"]
    ).median()

    carpet_to_built_up_ratio = (
        all_present_df["carpet_area"]
        / all_present_df["built_up_area"]
    ).median()

    logger.debug(
        "Median super-built-up to built-up ratio: %s",
        super_to_built_up_ratio,
    )
    logger.debug(
        "Median carpet to built-up ratio: %s",
        carpet_to_built_up_ratio,
    )

    # Case 1: super built-up and carpet area are available,
    # but built-up area is missing.
    sbc_mask = (
        df["super_built_up_area"].notna()
        & df["built_up_area"].isna()
        & df["carpet_area"].notna()
    )

    if sbc_mask.any():
        imputed_values = (
            (df.loc[sbc_mask, "super_built_up_area"] / super_to_built_up_ratio)
            + (df.loc[sbc_mask, "carpet_area"] / carpet_to_built_up_ratio)
        ) / 2

        df.loc[sbc_mask, "built_up_area"] = imputed_values.round()

        logger.debug(
            "Imputed built_up_area for %d rows using super-built-up and carpet areas",
            sbc_mask.sum(),
        )

    # Case 2: only super built-up area is available.
    sb_mask = (
        df["super_built_up_area"].notna()
        & df["built_up_area"].isna()
        & df["carpet_area"].isna()
    )

    if sb_mask.any():
        imputed_values = (
            df.loc[sb_mask, "super_built_up_area"] / super_to_built_up_ratio
        )

        df.loc[sb_mask, "built_up_area"] = imputed_values.round()

        logger.debug(
            "Imputed built_up_area for %d rows using super-built-up area",
            sb_mask.sum(),
        )

    # Case 3: only carpet area is available.
    c_mask = (
        df["super_built_up_area"].isna()
        & df["built_up_area"].isna()
        & df["carpet_area"].notna()
    )

    if c_mask.any():
        imputed_values = (
            df.loc[c_mask, "carpet_area"] / carpet_to_built_up_ratio
        )

        df.loc[c_mask, "built_up_area"] = imputed_values.round()

        logger.debug(
            "Imputed built_up_area for %d rows using carpet area",
            c_mask.sum(),
        )

    return df


def correct_area_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """
    For properties with unusually small built-up area but a high price,
    use the available `area` value as `built_up_area`.
    """
    anomaly_mask = (df["built_up_area"] < 2000) & (df["price"] > 2.5)

    if anomaly_mask.any():
        df.loc[anomaly_mask, "built_up_area"] = df.loc[anomaly_mask, "area"]

        logger.debug(
            "Corrected built_up_area using area for %d anomalous rows",
            anomaly_mask.sum(),
        )

    return df


def missing_value(df: pd.DataFrame) -> pd.DataFrame:
    logger.debug("Missing-value handling function started...")

    required_columns = [
        "price",
        "area",
        "areaWithType",
        "super_built_up_area",
        "built_up_area",
        "carpet_area",
        "floorNum",
        "facing",
        "agePossession",
        "sector",
        "property_type",
    ]

    missing_columns = [
        column for column in required_columns if column not in df.columns
    ]

    if missing_columns:
        raise KeyError(
            f"Missing required columns for missing-value handling: {missing_columns}"
        )

    df = df.copy()

    logger.debug(
        "Missing values before processing:\n%s",
        df[required_columns].isna().sum(),
    )

    # ---------------- Built-up area imputation ----------------

    df = impute_built_up_area(df)

    # Use the listing area when none of the parsed area fields could help.
    area_mask = df["built_up_area"].isna() & df["area"].notna()

    if area_mask.any():
        df.loc[area_mask, "built_up_area"] = df.loc[area_mask, "area"]

        logger.debug(
            "Imputed built_up_area from area for %d rows",
            area_mask.sum(),
        )

    # ---------------- Area anomaly correction ----------------

    df = correct_area_anomalies(df)

    # ---------------- Drop redundant or unusable columns ----------------

    columns_to_drop = [
        "area",
        "areaWithType",
        "super_built_up_area",
        "carpet_area",
        "area_room_ratio",
    ]

    existing_columns_to_drop = [
        column for column in columns_to_drop if column in df.columns
    ]

    if existing_columns_to_drop:
        df.drop(columns=existing_columns_to_drop, inplace=True)

        logger.debug("Dropped columns: %s", existing_columns_to_drop)

    # ---------------- Floor number imputation ----------------

    floor_num_missing_count = df["floorNum"].isna().sum()

    if floor_num_missing_count > 0:
        df["floorNum"] = df["floorNum"].fillna(2.0)

        logger.debug(
            "Filled %d missing floorNum values with 2.0",
            floor_num_missing_count,
        )

    # ---------------- Facing column ----------------

    if "facing" in df.columns:
        facing_missing_count = df["facing"].isna().sum()

        df.drop(columns=["facing"], inplace=True)

        logger.debug(
            "Dropped facing column. Missing values before dropping: %s",
            facing_missing_count,
        )

    # ---------------- Remove known bad record ----------------

    if 2536 in df.index:
        df.drop(index=[2536], inplace=True)

        logger.debug("Removed manually identified bad record with index 2536")
    else:
        logger.debug("Index 2536 was not present; no row removed")

    # ---------------- Age-possession imputation ----------------

    missing_age_mask = df["agePossession"].isna() | (
        df["agePossession"] == "Undefined"
    )
    undefined_before = int(missing_age_mask.sum())

    logger.debug(
        "Number of Undefined agePossession values before imputation: %s",
        undefined_before,
    )

    # Step 1: use mode by sector and property type.
    df["agePossession"] = df.apply(
        impute_age_possession_by_sector_and_property_type,
        axis=1,
        df=df,
    )

    # Step 2: use mode by sector only.
    df["agePossession"] = df.apply(
        impute_age_possession_by_sector,
        axis=1,
        df=df,
    )

    # Step 3: use mode by property type only.
    df["agePossession"] = df.apply(
        impute_age_possession_by_property_type,
        axis=1,
        df=df,
    )

    missing_age_mask = df["agePossession"].isna() | (
        df["agePossession"] == "Undefined"
    )
    undefined_after = int(missing_age_mask.sum())

    logger.debug(
        "Number of missing or Undefined agePossession values after imputation: %s",
        undefined_after,
    )

    if undefined_after > 0:
        valid_age_values = df.loc[
            df["agePossession"].notna()
            & (df["agePossession"] != "Undefined"),
            "agePossession",
        ].mode()

        if not valid_age_values.empty:
            df.loc[missing_age_mask, "agePossession"] = valid_age_values.iloc[0]
            logger.info(
                "Filled %d remaining agePossession values with the global mode",
                undefined_after,
            )
        else:
            df.loc[missing_age_mask, "agePossession"] = "Undefined"
            logger.warning(
                "No valid agePossession values were available; marked %d values "
                "as 'Undefined'",
                undefined_after,
            )

    remaining_missing = df.isna().sum()
    remaining_missing = remaining_missing[remaining_missing.gt(0)]
    if not remaining_missing.empty:
        raise ValueError(
            "Missing values remain after processing: "
            f"{remaining_missing.to_dict()}"
        )

    logger.debug("Missing values after processing:\n%s", df.isna().sum())
    logger.debug("Missing-value handling completed. Final shape: %s", df.shape)

    return df


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
        logger.debug("Raw dataframe columns: %s", list(df.columns))

        processed_df = missing_value(df=df)

        logger.debug("Processed dataframe shape: %s", processed_df.shape)
        logger.debug(
            "Processed dataframe columns: %s",
            list(processed_df.columns),
        )

        save_data(
            df=processed_df,
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

    data_path = (
        home_dir
        / "data"
        / "processed" 
        / "outlier_removed"
        / "gurgaon_outlier_removed.csv"
    )

    output_dir = home_dir / "data" / "processed" / "missing_value"
    output_path = output_dir / "gurgaon_missingvalue_removed.csv"

    logger.debug("Data path: %s", data_path)
    logger.debug("Output directory: %s", output_dir)
    logger.debug("Output path: %s", output_path)

    main(
        data_path=str(data_path),
        output_path=str(output_path),
        output_dir=str(output_dir),
    )