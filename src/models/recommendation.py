import ast
import logging
import os
import pathlib
import re
import sys
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler


# ---------------- Logging setup ----------------

def setup_logging(level: int = logging.DEBUG) -> logging.Logger:
    """Configure console logging for the property-recommendation stage."""
    logger = logging.getLogger("property_recommendation")
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

UNIT_CONFIGURATIONS = (
    "1 BHK",
    "2 BHK",
    "3 BHK",
    "4 BHK",
    "5 BHK",
    "6 BHK",
    "1 RK",
    "Land",
)

SIMILARITY_WEIGHTS = {
    "facilities": 30,
    "property_details": 20,
    "location": 8,
}

REQUIRED_COLUMNS = (
    "PropertyName",
    "TopFacilities",
    "PriceDetails",
    "LocationAdvantages",
)


def load_data(data_path: str) -> pd.DataFrame:
    """Load apartment metadata and validate columns used by the notebook."""
    logger.debug("Loading apartment metadata from: %s", data_path)

    if not os.path.isfile(data_path):
        raise FileNotFoundError(f"Input CSV file does not exist: {data_path}")

    try:
        df = pd.read_csv(data_path)
    except Exception:
        logger.exception("Failed to load apartment metadata: %s", data_path)
        raise

    missing_columns = [
        column for column in REQUIRED_COLUMNS if column not in df.columns
    ]
    if missing_columns:
        raise KeyError(
            f"Missing required apartment metadata columns: {missing_columns}"
        )

    # The source CSV contains a repeated header row at index 22; the notebook
    # removed it with drop(22). Filter by content so this remains safe if rows
    # are reordered in a later version of the input.
    property_names = df["PropertyName"].fillna("").astype(str).str.strip()
    repeated_header_mask = property_names.str.casefold().eq("propertyname")
    repeated_header_count = int(repeated_header_mask.sum())
    if repeated_header_count:
        df = df.loc[~repeated_header_mask].copy()
        logger.info("Removed %d repeated header row(s)", repeated_header_count)

    df["PropertyName"] = df["PropertyName"].astype(str).str.strip()
    df = df.loc[df["PropertyName"].ne("")].copy()

    if df.empty:
        raise ValueError("No valid property records remain in the input CSV")

    duplicate_names = df.loc[
        df["PropertyName"].duplicated(keep=False),
        "PropertyName",
    ].unique()
    if len(duplicate_names):
        raise ValueError(
            "PropertyName values must be unique to map recommendations "
            f"unambiguously. Duplicates: {duplicate_names[:10].tolist()}"
        )

    df.reset_index(drop=True, inplace=True)
    logger.info("Apartment metadata loaded. Shape: %s", df.shape)
    return df


def parse_literal_mapping(value: Any) -> dict[str, Any]:
    """Parse a Python-literal or JSON-like string into a dictionary."""
    if isinstance(value, dict):
        return value

    if not isinstance(value, str) or not value.strip():
        return {}

    try:
        parsed = ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return {}

    return parsed if isinstance(parsed, dict) else {}


def parse_facility_list(value: Any) -> list[str]:
    """Read facilities from the list-like CSV field used by the notebook."""
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]

    if not isinstance(value, str) or not value.strip():
        return []

    try:
        parsed = ast.literal_eval(value)
    except (ValueError, SyntaxError):
        parsed = None

    if isinstance(parsed, (list, tuple, set)):
        return [str(item).strip() for item in parsed if str(item).strip()]

    # Preserve the notebook's simple quoted-item parsing for malformed but
    # recoverable strings without executing arbitrary input.
    return [
        item.strip()
        for item in re.findall(r"'(.*?)'", value)
        if item.strip()
    ]


def parse_area_range(value: Any) -> tuple[float | None, float | None]:
    """Extract low/high area values from an area string."""
    if not isinstance(value, str):
        return None, None

    numbers = re.findall(r"\d[\d,]*(?:\.\d+)?", value)
    if not numbers:
        return None, None

    parsed_numbers = [
        float(number.replace(",", ""))
        for number in numbers[:2]
    ]
    if len(parsed_numbers) == 1:
        return parsed_numbers[0], parsed_numbers[0]
    return parsed_numbers[0], parsed_numbers[1]


def parse_price_value(value: str, inherited_unit: str) -> float | None:
    """Convert one price endpoint to crore rupees."""
    match = re.search(
        r"(\d[\d,]*(?:\.\d+)?)\s*(crores?|cr|lakhs?|l)?",
        value,
        flags=re.IGNORECASE,
    )
    if not match:
        return None

    amount = float(match.group(1).replace(",", ""))
    unit = (match.group(2) or inherited_unit).casefold()
    if unit.startswith("l"):
        amount /= 100
    return amount


def parse_price_range(value: Any) -> tuple[float | None, float | None]:
    """Extract low/high prices and normalize both values to crore rupees."""
    if not isinstance(value, str) or not value.strip():
        return None, None

    # A range's unit is sometimes written only on its final endpoint.
    unit_matches = re.findall(
        r"\b(crores?|cr|lakhs?|l)\b",
        value,
        flags=re.IGNORECASE,
    )
    inherited_unit = unit_matches[-1] if unit_matches else ""
    endpoints = re.split(r"\s*-\s*", value.strip(), maxsplit=1)

    low = parse_price_value(endpoints[0], inherited_unit)
    high = (
        parse_price_value(endpoints[1], inherited_unit)
        if len(endpoints) == 2
        else low
    )
    return low, high


def extract_property_details(price_details: Any) -> dict[str, Any]:
    """Parse per-unit type, area, and price features from PriceDetails."""
    detail_map = parse_literal_mapping(price_details)
    extracted: dict[str, Any] = {}

    for configuration, details in detail_map.items():
        if not isinstance(details, dict):
            continue

        extracted[f"building type_{configuration}"] = details.get(
            "building_type"
        )
        area_low, area_high = parse_area_range(details.get("area"))
        price_low, price_high = parse_price_range(details.get("price-range"))

        extracted[f"area low {configuration}"] = area_low
        extracted[f"area high {configuration}"] = area_high
        extracted[f"price low {configuration}"] = price_low
        extracted[f"price high {configuration}"] = price_high

    return extracted


def build_property_detail_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build and standardize per-unit property-detail features."""
    detail_records = []

    for row in df.itertuples(index=False):
        price_details = getattr(row, "PriceDetails")
        parsed_details = extract_property_details(price_details)
        detail_record: dict[str, Any] = {"PropertyName": row.PropertyName}

        for configuration in UNIT_CONFIGURATIONS:
            detail_record[
                f"building type_{configuration}"
            ] = parsed_details.get(f"building type_{configuration}")
            detail_record[f"area low {configuration}"] = parsed_details.get(
                f"area low {configuration}"
            )
            detail_record[f"area high {configuration}"] = parsed_details.get(
                f"area high {configuration}"
            )
            detail_record[f"price low {configuration}"] = parsed_details.get(
                f"price low {configuration}"
            )
            detail_record[f"price high {configuration}"] = parsed_details.get(
                f"price high {configuration}"
            )

        detail_records.append(detail_record)

    details_df = pd.DataFrame(detail_records).set_index("PropertyName")
    categorical_columns = details_df.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()
    encoded_df = pd.get_dummies(
        details_df,
        columns=categorical_columns,
        drop_first=True,
        dtype=float,
    )
    encoded_df = encoded_df.fillna(0).astype(float)

    if encoded_df.empty or encoded_df.shape[1] == 0:
        raise ValueError("Property-detail parsing produced no usable features")

    scaled_features = StandardScaler().fit_transform(encoded_df)
    return pd.DataFrame(
        scaled_features,
        columns=encoded_df.columns,
        index=encoded_df.index,
    )


def distance_to_meters(distance: Any) -> float | None:
    """Convert distance to meters; unsupported units remain missing."""
    if not isinstance(distance, str):
        return None

    match = re.search(r"\d[\d,]*(?:\.\d+)?", distance)
    if not match:
        return None

    value = float(match.group(0).replace(",", ""))
    normalized = distance.casefold()

    if (
        "km" in normalized
        or "kilometer" in normalized
        or "kilometre" in normalized
    ):
        return value * 1000
    if "meter" in normalized or re.search(r"\bm\b", normalized):
        return value
    return None


def build_location_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build and standardize the notebook's nearby-location distance matrix."""
    location_records = []

    for property_name, raw_advantages in zip(
        df["PropertyName"],
        df["LocationAdvantages"],
    ):
        advantages = parse_literal_mapping(raw_advantages)
        distances = {
            str(location): distance_to_meters(distance)
            for location, distance in advantages.items()
        }
        location_records.append(
            {"PropertyName": property_name, **distances}
        )

    location_df = pd.DataFrame(location_records).set_index("PropertyName")
    if location_df.empty or location_df.shape[1] == 0:
        raise ValueError("No location distances could be parsed")

    # The notebook used 54 km as the missing-location sentinel.
    location_df = location_df.fillna(54000).astype(float)
    scaled_features = StandardScaler().fit_transform(location_df)
    return pd.DataFrame(
        scaled_features,
        columns=location_df.columns,
        index=location_df.index,
    )


def build_recommendation_model(df: pd.DataFrame) -> dict[str, Any]:
    """
    Create the three weighted content-similarity matrices from the notebook.

    Weights (30/20/8) favor shared facilities, then unit/pricing details, then
    shared travel distances. Keeping the component matrices in the artifact
    makes it possible to tune this weighting later without re-parsing the CSV.
    """
    missing_columns = [
        column for column in REQUIRED_COLUMNS if column not in df.columns
    ]
    if missing_columns:
        raise KeyError(
            f"Missing required columns for recommendation modeling: "
            f"{missing_columns}"
        )

    properties = df.copy().reset_index(drop=True)
    property_names = properties["PropertyName"].astype(str).tolist()
    if len(set(property_names)) != len(property_names):
        raise ValueError("PropertyName values must be unique")

    facility_text = (
        properties["TopFacilities"].apply(parse_facility_list).apply(" ".join)
    )
    if not facility_text.str.strip().any():
        raise ValueError("No facility text could be parsed from TopFacilities")

    tfidf_vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
    )
    try:
        tfidf_matrix = tfidf_vectorizer.fit_transform(facility_text)
    except ValueError as error:
        raise ValueError(
            "TopFacilities did not contain usable text for TF-IDF"
        ) from error

    facilities_similarity = cosine_similarity(tfidf_matrix)

    detail_features = build_property_detail_features(properties)
    location_features = build_location_features(properties)

    # Align to the exact catalog row order before calculating each matrix.
    detail_features = detail_features.loc[property_names]
    location_features = location_features.loc[property_names]
    details_similarity = cosine_similarity(detail_features)
    location_similarity = cosine_similarity(location_features)

    combined_similarity = (
        SIMILARITY_WEIGHTS["facilities"] * facilities_similarity
        + SIMILARITY_WEIGHTS["property_details"] * details_similarity
        + SIMILARITY_WEIGHTS["location"] * location_similarity
    )

    matrix_shape = (len(property_names), len(property_names))
    for name, matrix in (
        ("facilities", facilities_similarity),
        ("property details", details_similarity),
        ("location", location_similarity),
        ("combined", combined_similarity),
    ):
        if matrix.shape != matrix_shape or not np.isfinite(matrix).all():
            raise ValueError(
                f"The {name} similarity matrix is invalid: {matrix.shape}"
            )

    logger.info(
        "Built recommender for %d properties with %d similarity components",
        len(property_names),
        3,
    )

    return {
        "format_version": 1,
        "property_names": property_names,
        "similarity_weights": SIMILARITY_WEIGHTS.copy(),
        "facilities_similarity": facilities_similarity,
        "property_details_similarity": details_similarity,
        "location_similarity": location_similarity,
        "combined_similarity": combined_similarity,
    }


def save_data(
    recommendation_model: dict[str, Any],
    properties: pd.DataFrame,
    output_dir: str,
) -> dict[str, str]:
    """Persist the recommender bundle and readable property catalog."""
    if recommendation_model is None or properties is None:
        raise ValueError(
            "Cannot save recommendation artifacts because data is None"
        )

    try:
        os.makedirs(output_dir, exist_ok=True)
        model_path = os.path.join(output_dir, "recommendation_model.joblib")
        catalog_path = os.path.join(output_dir, "property_catalog.csv")

        joblib.dump(recommendation_model, model_path)
        properties.to_csv(catalog_path, index=False)

        logger.info("Recommendation model saved to: %s", model_path)
        logger.info("Property catalog saved to: %s", catalog_path)
        return {"model": model_path, "catalog": catalog_path}
    except Exception:
        logger.exception(
            "Failed to save recommendation artifacts to %s",
            output_dir,
        )
        raise


def load_recommender(model_dir: str) -> dict[str, Any]:
    """Load and validate a recommendation artifact bundle."""
    model_path = os.path.join(model_dir, "recommendation_model.joblib")
    if not os.path.isfile(model_path):
        raise FileNotFoundError(
            f"Recommendation model artifact does not exist: {model_path}"
        )

    recommendation_model = joblib.load(model_path)
    required_keys = {
        "format_version",
        "property_names",
        "combined_similarity",
    }
    missing_keys = required_keys.difference(recommendation_model)
    if missing_keys:
        raise ValueError(
            "Recommendation artifact is missing fields: "
            f"{sorted(missing_keys)}"
        )

    return recommendation_model


def recommend_properties(
    property_name: str,
    recommendation_model: dict[str, Any],
    top_n: int = 10,
) -> pd.DataFrame:
    """Return the top-N similar properties, excluding the query property."""
    if top_n < 1:
        raise ValueError(f"top_n must be at least 1; received {top_n}")

    property_names = recommendation_model["property_names"]
    normalized_names = [name.strip().casefold() for name in property_names]
    query = property_name.strip().casefold()
    matching_indices = [
        index for index, name in enumerate(normalized_names) if name == query
    ]

    if not matching_indices:
        raise KeyError(
            "Property not found in recommendation catalog: "
            f"{property_name}"
        )
    if len(matching_indices) > 1:
        raise ValueError(f"Property name is ambiguous: {property_name}")

    query_index = matching_indices[0]
    similarity_scores = np.asarray(
        recommendation_model["combined_similarity"]
    )[query_index]

    # Exclude the selected property itself, then sort by the notebook's
    # weighted cosine similarity from highest to lowest.
    ranked_indices = np.argsort(similarity_scores)[::-1]
    ranked_indices = [
        int(index) for index in ranked_indices if index != query_index
    ][:top_n]

    return pd.DataFrame(
        {
            "PropertyName": [
                property_names[index] for index in ranked_indices
            ],
            "SimilarityScore": [
                float(similarity_scores[index]) for index in ranked_indices
            ],
        }
    )


def main(data_path: str, output_dir: str) -> None:
    """Run the recommendation artifact-building pipeline."""
    try:
        logger.info("Starting property-recommendation pipeline")
        properties = load_data(data_path=data_path)
        recommendation_model = build_recommendation_model(properties)
        save_data(
            recommendation_model=recommendation_model,
            properties=properties,
            output_dir=output_dir,
        )
        logger.info("Property-recommendation pipeline completed successfully")
    except Exception:
        logger.exception("Property-recommendation pipeline failed")
        raise


if __name__ == "__main__":
    current_file = pathlib.Path(__file__)
    home_dir = current_file.parent.parent.parent

    data_path = home_dir / "data" / "raw" / "appartments.csv"
    output_dir = home_dir / "model" / "recommendation"

    logger.info("Current file: %s", current_file)
    logger.info("Home directory: %s", home_dir)
    logger.debug("Input data path: %s", data_path)
    logger.debug("Output directory: %s", output_dir)

    main(
        data_path=str(data_path),
        output_dir=str(output_dir),
    )
