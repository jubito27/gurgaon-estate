import os
import re
import sys
import ast
import pathlib
import logging

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer


# ---------------- Logging setup ----------------

def setup_logging(level: int = logging.DEBUG) -> logging.Logger:
    logger = logging.getLogger("data_preprocessing_houses")
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


def load_data(data_path: str, apartment_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    logger.debug("Loading data function started...")
    logger.debug("Loading data")

    if not os.path.isfile(data_path):
        logger.error("Input CSV file does not exist: %s", data_path)
        raise FileNotFoundError(f"Input CSV file does not exist: {data_path}")

    if not os.path.isfile(apartment_path):
        logger.error("Apartment CSV file does not exist: %s", apartment_path)
        raise FileNotFoundError(f"Apartment CSV file does not exist: {apartment_path}")

    try:
        df = pd.read_csv(data_path)
        app_df = pd.read_csv(apartment_path)

        logger.debug("Main data loaded successfully. Shape: %s", df.shape)
        logger.debug("Apartment data loaded successfully. Shape: %s", app_df.shape)

        return df, app_df

    except Exception:
        logger.exception("Failed to load data from %s or %s", data_path, apartment_path)
        raise


def get_super_built_up_area(text) -> float | None:
    if not isinstance(text, str):
        return None

    match = re.search(r"Super Built up area\s*(\d+\.?\d*)", text, flags=re.IGNORECASE)
    return float(match.group(1)) if match else None


def get_area(text, area_type: str) -> float | None:
    if not isinstance(text, str):
        return None

    pattern = re.escape(area_type) + r"\s*:\s*(\d+\.?\d*)"
    match = re.search(pattern, text, flags=re.IGNORECASE)
    return float(match.group(1)) if match else None


def convert_to_sqft(text, area_value) -> float | None:
    if area_value is None or not isinstance(text, str):
        return area_value

    pattern = re.escape(str(area_value)) + r"\s*\((\d+\.?\d*)\s*sq\.?m\.?\)"
    match = re.search(pattern, text, flags=re.IGNORECASE)

    if match:
        return float(match.group(1)) * 10.7639

    return area_value


def extract_plot_area(area_with_type) -> float | None:
    if not isinstance(area_with_type, str):
        return None

    match = re.search(r"Plot area\s*(\d+\.?\d*)", area_with_type, flags=re.IGNORECASE)
    return float(match.group(1)) if match else None


def convert_scale(row) -> float | None:
    area = row["area"]
    built_up_area = row["built_up_area"]

    if pd.isna(area) or pd.isna(built_up_area) or built_up_area == 0:
        return built_up_area

    ratio = area / built_up_area

    if round(ratio) == 9:
        return built_up_area * 9
    if round(ratio) == 11:
        return built_up_area * 10.7

    return built_up_area


def categorize_age_possession(value) -> str:
    if pd.isna(value):
        return "Undefined"

    value = str(value)

    if any(
        phrase in value
        for phrase in (
            "0 to 1 Year Old",
            "Within 6 months",
            "Within 3 months",
        )
    ):
        return "New Property"

    if "1 to 5 Year Old" in value:
        return "Relatively New"

    if "5 to 10 Year Old" in value:
        return "Moderately Old"

    if "10+ Year Old" in value:
        return "Old Property"

    if "Under Construction" in value or "By" in value:
        return "Under Construction"

    try:
        int(value.split()[-1])
        return "Under Construction"
    except (ValueError, IndexError):
        return "Undefined"


def get_furnishing_count(details, furnishing: str) -> int:
    if not isinstance(details, str):
        return 0

    if f"No {furnishing}" in details:
        return 0

    pattern = re.compile(rf"(\d+)\s+{re.escape(furnishing)}", flags=re.IGNORECASE)
    match = pattern.search(details)

    if match:
        return int(match.group(1))

    if re.search(rf"\b{re.escape(furnishing)}\b", details, flags=re.IGNORECASE):
        return 1

    return 0


def normalize_feature_name(name: str) -> str:
    """Make feature-name matching insensitive to case, spacing and punctuation."""
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


LUXURY_WEIGHTS = {
    "24/7 Power Backup": 8,
    "24/7 Water Supply": 4,
    "24x7 Security": 7,
    "ATM": 4,
    "Aerobics Centre": 6,
    "Airy Rooms": 8,
    "Amphitheatre": 7,
    "Badminton Court": 7,
    "Banquet Hall": 8,
    "Bar/Chill-Out Lounge": 9,
    "Barbecue": 7,
    "Basketball Court": 7,
    "Billiards": 7,
    "Bowling Alley": 8,
    "Business Lounge": 9,
    "CCTV Camera Security": 8,
    "Cafeteria": 6,
    "Car Parking": 6,
    "Card Room": 6,
    "Centrally Air Conditioned": 9,
    "Changing Area": 6,
    "Children's Play Area": 7,
    "Cigar Lounge": 9,
    "Clinic": 5,
    "Club House": 9,
    "Concierge Service": 9,
    "Conference room": 8,
    "Creche/Day care": 7,
    "Cricket Pitch": 7,
    "Doctor on Call": 6,
    "Earthquake Resistant": 5,
    "Entrance Lobby": 7,
    "False Ceiling Lighting": 6,
    "Feng Shui / Vaastu Compliant": 5,
    "Fire Fighting Systems": 8,
    "Fitness Centre / GYM": 8,
    "Flower Garden": 7,
    "Food Court": 6,
    "Foosball": 5,
    "Football": 7,
    "Fountain": 7,
    "Gated Community": 7,
    "Golf Course": 10,
    "Grocery Shop": 6,
    "Gymnasium": 8,
    "High Ceiling Height": 8,
    "High Speed Elevators": 8,
    "Infinity Pool": 9,
    "Intercom Facility": 7,
    "Internal Street Lights": 6,
    "Internet/wi-fi connectivity": 7,
    "Jacuzzi": 9,
    "Jogging Track": 7,
    "Landscape Garden": 8,
    "Laundry": 6,
    "Lawn Tennis Court": 8,
    "Library": 8,
    "Lounge": 8,
    "Low Density Society": 7,
    "Maintenance Staff": 6,
    "Manicured Garden": 7,
    "Medical Centre": 5,
    "Milk Booth": 4,
    "Mini Theatre": 9,
    "Multipurpose Court": 7,
    "Multipurpose Hall": 7,
    "Natural Light": 8,
    "Natural Pond": 7,
    "Park": 8,
    "Party Lawn": 8,
    "Piped Gas": 7,
    "Pool Table": 7,
    "Power Back up Lift": 8,
    "Private Garden / Terrace": 9,
    "Property Staff": 7,
    "RO System": 7,
    "Rain Water Harvesting": 7,
    "Reading Lounge": 8,
    "Restaurant": 8,
    "Salon": 8,
    "Sauna": 9,
    "Security / Fire Alarm": 9,
    "Security Personnel": 9,
    "Separate entry for servant room": 8,
    "Sewage Treatment Plant": 6,
    "Shopping Centre": 7,
    "Skating Rink": 7,
    "Solar Lighting": 6,
    "Solar Water Heating": 7,
    "Spa": 9,
    "Spacious Interiors": 9,
    "Squash Court": 8,
    "Steam Room": 9,
    "Sun Deck": 8,
    "Swimming Pool": 8,
    "Temple": 5,
    "Theatre": 9,
    "Toddler Pool": 7,
    "Valet Parking": 9,
    "Video Door Security": 9,
    "Visitor Parking": 7,
    "Water Softener Plant": 7,
    "Water Storage": 7,
    "Water purifier": 7,
    "Yoga/Meditation Area": 7,
}


def parse_feature_list(value) -> list:
    if not isinstance(value, str):
        return []

    value = value.strip()

    if not value.startswith("["):
        return []

    try:
        parsed = ast.literal_eval(value)

        if isinstance(parsed, list):
            return [str(item).strip() for item in parsed if str(item).strip()]

        return []

    except (ValueError, SyntaxError):
        return []


def feature_engineering(df: pd.DataFrame, app_df: pd.DataFrame) -> pd.DataFrame:
    logger.debug("Feature engineering function started...")
    logger.debug("Processing...")

    required_columns = [
        "areaWithType",
        "additionalRoom",
        "agePossession",
        "furnishDetails",
        "features",
        "society",
        "area",
    ]

    missing_columns = [column for column in required_columns if column not in df.columns]

    if missing_columns:
        raise KeyError(f"Missing required columns in main data: {missing_columns}")

    if "PropertyName" not in app_df.columns or "TopFacilities" not in app_df.columns:
        raise KeyError(
            "Apartment data must contain 'PropertyName' and 'TopFacilities' columns."
        )

    # ---------------- Area extraction ----------------

    df["super_built_up_area"] = df["areaWithType"].apply(get_super_built_up_area)
    df["super_built_up_area"] = df.apply(
        lambda row: convert_to_sqft(row["areaWithType"], row["super_built_up_area"]),
        axis=1,
    )

    df["built_up_area"] = df["areaWithType"].apply(
        lambda text: get_area(text, "Built Up area")
    )
    df["built_up_area"] = df.apply(
        lambda row: convert_to_sqft(row["areaWithType"], row["built_up_area"]),
        axis=1,
    )

    df["carpet_area"] = df["areaWithType"].apply(
        lambda text: get_area(text, "Carpet area")
    )
    df["carpet_area"] = df.apply(
        lambda row: convert_to_sqft(row["areaWithType"], row["carpet_area"]),
        axis=1,
    )

    all_area_missing = (
        df["super_built_up_area"].isna()
        & df["built_up_area"].isna()
        & df["carpet_area"].isna()
    )

    if all_area_missing.any():
        plot_area = df.loc[all_area_missing, "areaWithType"].apply(extract_plot_area)
        converted_plot_area = df[all_area_missing].apply(convert_scale, axis=1)

        df.loc[all_area_missing, "built_up_area"] = converted_plot_area

        logger.debug(
            "Imputed built-up area from plot area for %d rows",
            all_area_missing.sum(),
        )

    # ---------------- Additional rooms ----------------

    additional_room_columns = [
        "study room",
        "servant room",
        "store room",
        "pooja room",
        "others",
    ]

    additional_room_text = df["additionalRoom"].fillna("").astype(str).str.lower()

    for column in additional_room_columns:
        df[column] = additional_room_text.str.contains(
            re.escape(column), regex=True
        ).astype(int)

    # ---------------- Age / possession ----------------

    df["agePossession"] = df["agePossession"].apply(categorize_age_possession)

    # ---------------- Furnishing details ----------------

    all_furnishings = []

    for detail in df["furnishDetails"].dropna():
        if not isinstance(detail, str):
            continue

        cleaned = (
            detail.replace("[", "")
            .replace("]", "")
            .replace("'", "")
            .replace('"', "")
        )

        all_furnishings.extend(
            item.strip() for item in cleaned.split(",") if item.strip()
        )

    unique_furnishings = sorted(set(all_furnishings))

    columns_to_include = sorted(
        {
            re.sub(r"^No\s+|\d+", "", furnishing).strip()
            for furnishing in unique_furnishings
        }
    )
    columns_to_include = [item for item in columns_to_include if item]

    if columns_to_include:
        for furnishing in columns_to_include:
            df[furnishing] = df["furnishDetails"].apply(
                lambda value, furnishing=furnishing: get_furnishing_count(
                    value, furnishing
                )
            )

        furnishings_df = df[columns_to_include].fillna(0)

        scaler = StandardScaler()
        scaled_furnishing_data = scaler.fit_transform(furnishings_df)

        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        df["furnishing_type"] = kmeans.fit_predict(scaled_furnishing_data)

        logger.debug("Created %d furnishing columns", len(columns_to_include))
    else:
        df["furnishing_type"] = 0
        logger.warning("No furnishing details found; furnishing_type set to 0")

    # ---------------- Apartment features ----------------

    app_df["PropertyName"] = app_df["PropertyName"].astype(str).str.lower().str.strip()
    app_df = app_df.drop_duplicates(subset=["PropertyName"])

    missing_features_mask = df["features"].isna()

    if missing_features_mask.any():
        apartment_features = (
            df.loc[missing_features_mask]
            .merge(
                app_df[["PropertyName", "TopFacilities"]],
                left_on=df.loc[missing_features_mask, "society"]
                .astype(str)
                .str.lower()
                .str.strip(),
                right_on="PropertyName",
                how="left",
            )["TopFacilities"]
        )

        df.loc[missing_features_mask, "features"] = apartment_features.to_numpy()

    df["features_list"] = df["features"].apply(parse_feature_list)

    # Normalize names so 'PowerBack up Lift' and 'Power Back up Lift' match.
    df["features_list"] = df["features_list"].apply(
        lambda values: [normalize_feature_name(value) for value in values]
    )

    mlb = MultiLabelBinarizer()
    feature_matrix = mlb.fit_transform(df["features_list"])

    feature_columns = [normalize_feature_name(column) for column in mlb.classes_]
    features_binary_df = pd.DataFrame(
        feature_matrix, columns=feature_columns, index=df.index
    )

    # ---------------- Luxury score ----------------

    normalized_weights = {
        normalize_feature_name(name): weight
        for name, weight in LUXURY_WEIGHTS.items()
    }

    available_weighted_features = [
        column for column in features_binary_df.columns if column in normalized_weights
    ]

    if available_weighted_features:
        luxury_score = (
            features_binary_df[available_weighted_features]
            .mul(
                pd.Series(
                    {
                        column: normalized_weights[column]
                        for column in available_weighted_features
                    }
                )
            )
            .sum(axis=1)
        )
    else:
        luxury_score = pd.Series(0, index=df.index)

    df["luxury_score"] = luxury_score

    unmatched_features = set(features_binary_df.columns) - set(normalized_weights)

    if unmatched_features:
        logger.debug(
            "%d feature columns had no luxury weight and were ignored",
            len(unmatched_features),
        )

    # ---------------- Cleanup ----------------

    columns_to_drop = [
        column
        for column in [
            "nearbyLocations",
            "furnishDetails",
            "features",
            "features_list",
            "additionalRoom",
        ]
        if column in df.columns
    ]

    df.drop(columns=columns_to_drop, inplace=True)

    logger.debug("Feature engineering completed. Final shape: %s", df.shape)
    logger.debug("Final columns: %s", list(df.columns))

    return df


def save_data(df: pd.DataFrame, output_path: str, output_dir: str) -> None:
    logger.debug("save_data function started")

    if df is None:
        raise ValueError("Cannot save data because the feature-engineered DataFrame is None.")

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
    apartment_path: str,
    output_path: str,
    output_dir: str,
) -> None:
    try:
        logger.info("Main function execution started")

        df, app_df = load_data(
            data_path=data_path,
            apartment_path=apartment_path,
        )

        logger.debug(
            "Raw dataframe shapes - main: %s, apartments: %s",
            df.shape,
            app_df.shape,
        )

        df = feature_engineering(df=df, app_df=app_df)

        logger.debug("Feature engineering completed")

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

    data_path = home_dir / "data" / "processed" / "cleaned" / "gurgaon.csv"
    apartment_path = home_dir / "data" / "raw" / "appartments.csv"
    output_dir = home_dir / "data" / "interim" / "v1"
    output_path = output_dir / "gurgaon_v1.csv"

    logger.debug("Data path: %s", data_path)
    logger.debug("Apartment path: %s", apartment_path)
    logger.debug("Output directory: %s", output_dir)
    logger.debug("Output path: %s", output_path)

    main(
        data_path=str(data_path),
        apartment_path=str(apartment_path),
        output_path=str(output_path),
        output_dir=str(output_dir),
    )