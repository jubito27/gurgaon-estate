import ast
import base64
import io
import logging
import math
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
import seaborn as sns
from flask import Flask, Response, jsonify, render_template, request
from wordcloud import WordCloud


# ---------------- Logging and paths ----------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("gurgaon_flask_app")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_FEATURES_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "feature_selection"
    / "gurgaon_properties_post_feature_selection_v2.csv"
)
RAW_FEATURES_PATH = (
    RAW_DATA_DIR / "gurgaon_properties_post_feature_selection_v2.csv"
)
ANALYTICS_DATA_PATH = RAW_DATA_DIR / "data_viz1.csv"
PROPERTY_CATALOG_PATH = (
    PROJECT_ROOT / "model" / "recommendation" / "property_catalog.csv"
)
APARTMENT_DATA_PATH = RAW_DATA_DIR / "appartments.csv"
PROPERTY_FEATURES_PATH = RAW_DATA_DIR / "gurgaon_properties.csv"
PIPELINE_PATH = PROJECT_ROOT / "model" / "pipeline.pkl"
RECOMMENDATION_MODEL_PATH = (
    PROJECT_ROOT
    / "model"
    / "recommendation"
    / "recommendation_model.joblib"
)

CARD_BG_COLOR = "#1e293b"
PREDICTION_MARGIN_CR = 0.22

pio.templates.default = "plotly_dark"
plt.style.use("dark_background")

app = Flask(__name__)


def load_csv(path: Path, required_columns: tuple[str, ...]) -> pd.DataFrame:
    """Load a project CSV and report missing files or required columns clearly."""
    if not path.is_file():
        raise FileNotFoundError(f"Required project data file not found: {path}")

    df = pd.read_csv(path)
    missing_columns = [
        column for column in required_columns if column not in df.columns
    ]
    if missing_columns:
        raise KeyError(
            f"{path} is missing required columns: {missing_columns}"
        )
    return df


def sorted_values(series: pd.Series) -> list[Any]:
    """Return unique dropdown values in a stable order, including mixed types."""
    values = series.dropna().unique().tolist()
    return sorted(values, key=lambda value: str(value).casefold())


def load_apartment_data() -> pd.DataFrame:
    """Load the notebook's raw apartment catalog and remove its repeated header."""
    # The DVC recommendation stage publishes the catalog alongside its model.
    # Prefer that deployable artifact; retain the raw CSV as a local-dev fallback.
    source_path = (
        PROPERTY_CATALOG_PATH
        if PROPERTY_CATALOG_PATH.is_file()
        else APARTMENT_DATA_PATH
    )
    df = load_csv(
        source_path,
        (
            "PropertyName",
            "LocationAdvantages",
            "TopFacilities",
        ),
    )
    names = df["PropertyName"].fillna("").astype(str).str.strip()
    df = df.loc[names.str.casefold().ne("propertyname")].copy()
    df["PropertyName"] = df["PropertyName"].astype(str).str.strip()
    df = df.loc[df["PropertyName"].ne("")].reset_index(drop=True)

    if df.empty:
        raise ValueError(f"No valid apartment rows found in {APARTMENT_DATA_PATH}")
    return df


def parse_literal_dict(value: Any) -> dict[str, Any]:
    """Safely parse a dictionary stored as a CSV string."""
    if isinstance(value, dict):
        return value
    if not isinstance(value, str) or not value.strip():
        return {}

    try:
        parsed = ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def distance_to_meters(distance: Any) -> float | None:
    """Convert supported kilometer or meter strings into meters."""
    if not isinstance(distance, str):
        return None

    match = re.search(r"\d[\d,]*(?:\.\d+)?", distance)
    if not match:
        return None

    amount = float(match.group(0).replace(",", ""))
    normalized = distance.casefold()
    if "km" in normalized or "kilometer" in normalized or "kilometre" in normalized:
        return amount * 1000
    if "meter" in normalized or re.search(r"\bm\b", normalized):
        return amount
    return None


def build_location_frame(properties: pd.DataFrame) -> pd.DataFrame:
    """Build the landmark-distance table used by the radius-search endpoint."""
    rows = []
    for property_name, raw_advantages in zip(
        properties["PropertyName"],
        properties["LocationAdvantages"],
    ):
        advantages = parse_literal_dict(raw_advantages)
        rows.append(
            {
                "PropertyName": property_name,
                **{
                    str(location): distance_to_meters(distance)
                    for location, distance in advantages.items()
                },
            }
        )

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).set_index("PropertyName")


def build_wordcloud_text() -> str:
    """Use the original amenity lists as the word-cloud source."""
    if PROPERTY_FEATURES_PATH.is_file():
        features_df = load_csv(PROPERTY_FEATURES_PATH, ("features",))
        words: list[str] = []
        for value in features_df["features"].dropna():
            if not isinstance(value, str):
                continue
            try:
                parsed = ast.literal_eval(value)
            except (ValueError, SyntaxError):
                continue
            if isinstance(parsed, (list, tuple, set)):
                words.extend(str(item) for item in parsed if str(item).strip())
        if words:
            return " ".join(words)

    # Keep the visualization usable if the detailed property file is absent
    # or contains no parseable feature lists.
    words = []
    for value in apartment_df["TopFacilities"].dropna():
        if isinstance(value, str):
            try:
                parsed = ast.literal_eval(value)
            except (ValueError, SyntaxError):
                continue
            if isinstance(parsed, (list, tuple, set)):
                words.extend(str(item) for item in parsed if str(item).strip())
    return " ".join(words)


# Prefer the generated pipeline input; the raw v2 file supports local app use
# before DVC stages have generated the processed copy.
FEATURES_DATA_PATH = (
    PROCESSED_FEATURES_PATH
    if PROCESSED_FEATURES_PATH.is_file()
    else RAW_FEATURES_PATH
)
training_df = load_csv(
    FEATURES_DATA_PATH,
    (
        "property_type",
        "sector",
        "price",
        "bedRoom",
        "bathroom",
        "balcony",
        "agePossession",
        "built_up_area",
        "servant room",
        "store room",
        "furnishing_type",
        "luxury_category",
        "floor_category",
    ),
)

# The original exploratory notebook's data_viz1.csv is optional at deploy time.
# The selected-feature DVC artifact is sufficient for all non-geographic charts.
if ANALYTICS_DATA_PATH.is_file():
    analytics_df = load_csv(
        ANALYTICS_DATA_PATH,
        ("property_type", "sector", "price", "bedRoom", "built_up_area"),
    )
else:
    analytics_df = training_df.copy()
    logger.info(
        "Optional analytics CSV not found; using the DVC feature-selection data"
    )

for column in (
    "price",
    "price_per_sqft",
    "built_up_area",
    "latitude",
    "longitude",
    "bedRoom",
):
    if column in analytics_df.columns:
        analytics_df[column] = pd.to_numeric(
            analytics_df[column],
            errors="coerce",
        )

if "price_per_sqft" not in analytics_df.columns:
    area = pd.to_numeric(analytics_df["built_up_area"], errors="coerce")
    price = pd.to_numeric(analytics_df["price"], errors="coerce")
    analytics_df["price_per_sqft"] = np.where(
        area > 0,
        price * 10000 / area,
        np.nan,
    )

apartment_df = load_apartment_data()
location_df = build_location_frame(apartment_df)

if location_df.empty or location_df.shape[1] == 0:
    logger.warning("No parseable location distances were found in the raw catalog")


# ---------------- Cached model loading ----------------

@lru_cache(maxsize=1)
def load_prediction_pipeline():
    """Load the DVC-trained pipeline only when a prediction is requested."""
    if not PIPELINE_PATH.is_file():
        raise FileNotFoundError(
            "Trained prediction pipeline not found. Run "
            "`dvc repro 10_model_training` first: "
            f"{PIPELINE_PATH}"
        )

    try:
        import joblib
    except ImportError as error:
        raise RuntimeError(
            "joblib is required to load the prediction pipeline. "
            "Install project dependencies with `pip install -r requirements.txt`."
        ) from error

    pipeline = joblib.load(PIPELINE_PATH)
    if pipeline is None:
        raise ValueError(f"Loaded prediction pipeline is empty: {PIPELINE_PATH}")
    return pipeline


@lru_cache(maxsize=1)
def load_recommendation_model() -> dict[str, Any]:
    """Load and validate the DVC-built recommender artifact when needed."""
    if not RECOMMENDATION_MODEL_PATH.is_file():
        raise FileNotFoundError(
            "Recommendation model not found. Run "
            "`dvc repro 13_recommendation` first: "
            f"{RECOMMENDATION_MODEL_PATH}"
        )

    try:
        import joblib
    except ImportError as error:
        raise RuntimeError(
            "joblib is required to load the recommendation model. "
            "Install project dependencies with `pip install -r requirements.txt`."
        ) from error

    model = joblib.load(RECOMMENDATION_MODEL_PATH)
    required_keys = {"property_names", "combined_similarity"}
    if not isinstance(model, dict) or not required_keys.issubset(model):
        raise ValueError(
            "Recommendation artifact has an invalid format: "
            f"{RECOMMENDATION_MODEL_PATH}"
        )

    names = model["property_names"]
    similarity = np.asarray(model["combined_similarity"])
    expected_shape = (len(names), len(names))
    if similarity.shape != expected_shape or not np.isfinite(similarity).all():
        raise ValueError(
            "Recommendation similarity matrix has an invalid shape or values"
        )

    model["combined_similarity"] = similarity
    return model


def api_error(message: str, status_code: int):
    """Return a consistent API error object."""
    return jsonify({"status": "error", "message": message}), status_code


def get_json_object():
    """Reject missing or non-object JSON request bodies."""
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return None
    return data


# ---------------- Page and options APIs ----------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/health")
def health():
    """Lightweight container and load-balancer health check."""
    return jsonify({"status": "ok"}), 200


@app.route("/api/options", methods=["GET"])
def get_options():
    return jsonify(
        {
            "sectors": sorted_values(training_df["sector"]),
            "bedrooms": sorted_values(training_df["bedRoom"]),
            "bathrooms": sorted_values(training_df["bathroom"]),
            "balconies": sorted_values(training_df["balcony"]),
            "property_age": sorted_values(training_df["agePossession"]),
            "furnishing_type": sorted_values(training_df["furnishing_type"]),
            "luxury_category": sorted_values(training_df["luxury_category"]),
            "floor_category": sorted_values(training_df["floor_category"]),
            "locations": sorted(location_df.columns.tolist()),
            "apartments": apartment_df["PropertyName"].tolist(),
            "analytics_sectors": ["overall"]
            + sorted_values(analytics_df["sector"]),
        }
    )


# ---------------- Price prediction API ----------------

@app.route("/api/predict", methods=["POST"])
def predict_price():
    data = get_json_object()
    if data is None:
        return api_error("Request body must be a JSON object", 400)

    required_fields = (
        "property_type",
        "sector",
        "bedrooms",
        "bathroom",
        "balcony",
        "property_age",
        "built_up_area",
        "servant_room",
        "store_room",
        "furnishing_type",
        "luxury_category",
        "floor_category",
    )
    missing_fields = [
        field for field in required_fields if field not in data
    ]
    if missing_fields:
        return api_error(f"Missing prediction fields: {missing_fields}", 400)

    try:
        category_fields = {
            "property_type": "property_type",
            "sector": "sector",
            "balcony": "balcony",
            "property_age": "agePossession",
            "luxury_category": "luxury_category",
            "floor_category": "floor_category",
        }
        category_values = {}
        for request_field, data_column in category_fields.items():
            value = data[request_field]
            if value is None or not str(value).strip():
                raise ValueError(f"{request_field} must not be empty")
            value = str(value)
            allowed_values = {
                str(category)
                for category in training_df[data_column].dropna().unique()
            }
            if value not in allowed_values:
                raise ValueError(
                    f"Unsupported {request_field}: {value}"
                )
            category_values[request_field] = value

        bedroom_count = float(data["bedrooms"])
        bathroom_count = float(data["bathroom"])
        built_up_area = float(data["built_up_area"])
        servant_room = float(data["servant_room"])
        store_room = float(data["store_room"])
        furnishing_type = float(data["furnishing_type"])

        numeric_values = (
            bedroom_count,
            bathroom_count,
            built_up_area,
            servant_room,
            store_room,
            furnishing_type,
        )
        if not all(math.isfinite(value) for value in numeric_values):
            raise ValueError("Numeric prediction fields must be finite")
        if bedroom_count < 0 or bathroom_count < 0 or built_up_area <= 0:
            raise ValueError(
                "Bedrooms and bathrooms must be non-negative, and "
                "built-up area must be greater than zero"
            )
        if servant_room < 0 or store_room < 0:
            raise ValueError("Room counts must be non-negative")
        allowed_furnishing_types = pd.to_numeric(
            training_df["furnishing_type"],
            errors="coerce",
        ).dropna().unique()
        if not any(
            math.isclose(furnishing_type, float(value))
            for value in allowed_furnishing_types
        ):
            raise ValueError(
                f"Unsupported furnishing_type: {furnishing_type}"
            )

        input_df = pd.DataFrame(
            [
                {
                    "property_type": category_values["property_type"],
                    "sector": category_values["sector"],
                    "bedRoom": bedroom_count,
                    "bathroom": bathroom_count,
                    "balcony": category_values["balcony"],
                    "agePossession": category_values["property_age"],
                    "built_up_area": built_up_area,
                    "servant room": servant_room,
                    "store room": store_room,
                    "furnishing_type": furnishing_type,
                    "luxury_category": category_values["luxury_category"],
                    "floor_category": category_values["floor_category"],
                }
            ]
        )
    except (TypeError, ValueError) as error:
        return api_error(str(error), 400)

    try:
        pipeline = load_prediction_pipeline()
    except (FileNotFoundError, RuntimeError) as error:
        return api_error(str(error), 503)
    except Exception:
        logger.exception("Could not load prediction pipeline")
        return api_error("Could not load prediction pipeline; see server logs", 500)

    try:
        prediction = float(np.asarray(pipeline.predict(input_df)).reshape(-1)[0])
        if not math.isfinite(prediction):
            raise RuntimeError("The trained model returned a non-finite prediction")

        # The current train_model.py fits directly to `price` in crores.
        # Do not apply expm1 here; that would distort the raw-price prediction.
        base_price = max(prediction, 0.0)
        low = max(0.0, round(base_price - PREDICTION_MARGIN_CR, 2))
        high = round(base_price + PREDICTION_MARGIN_CR, 2)
        return jsonify({"status": "success", "low": low, "high": high})
    except Exception:
        logger.exception("Price prediction failed")
        return api_error("Price prediction failed; see server logs", 500)


# ---------------- Analytics APIs ----------------

def apply_transparent_bg(fig):
    """Keep charts consistent with the dashboard's dark card backgrounds."""
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def plot_response(fig) -> Response:
    return Response(fig.to_json(), mimetype="application/json")


@app.route("/api/analytics/geomap")
def analytics_geomap():
    # The checked-in DVC outputs do not include latitude/longitude coordinates.
    # Show a useful sector-level price chart instead of returning a broken API.
    if not {"latitude", "longitude"}.issubset(analytics_df.columns):
        sector_df = (
            analytics_df.groupby("sector", as_index=False)
            .agg(
                price_per_sqft=("price_per_sqft", "median"),
                listings=("price", "size"),
            )
            .dropna(subset=["price_per_sqft"])
            .sort_values("listings", ascending=False)
            .head(20)
        )
        if sector_df.empty:
            return api_error("No sector analytics are available", 404)
        fig = px.bar(
            sector_df,
            x="sector",
            y="price_per_sqft",
            color="listings",
            title="Price per Sq Ft by Sector (Top 20)",
            labels={
                "sector": "Sector",
                "price_per_sqft": "Median price per sq ft",
                "listings": "Listings",
            },
        )
        fig.update_layout(xaxis_tickangle=-35)
        return plot_response(apply_transparent_bg(fig))

    try:
        group_df = (
            analytics_df.groupby("sector", as_index=False)[
                [
                    "price",
                    "price_per_sqft",
                    "built_up_area",
                    "latitude",
                    "longitude",
                ]
            ]
            .mean()
            .dropna(
                subset=[
                    "latitude",
                    "longitude",
                    "built_up_area",
                    "price_per_sqft",
                ]
            )
        )
        group_df = group_df[group_df["built_up_area"] > 0]
        if group_df.empty:
            return api_error("No valid geographic analytics rows are available", 404)

        map_options = {
            "lat": "latitude",
            "lon": "longitude",
            "color": "price_per_sqft",
            "size": "built_up_area",
            "color_continuous_scale": px.colors.cyclical.IceFire,
            "zoom": 10,
            "center": {
                "lat": float(group_df["latitude"].mean()),
                "lon": float(group_df["longitude"].mean()),
            },
            "text": "sector",
            "hover_name": "sector",
        }
        # Plotly renamed scatter_mapbox to scatter_map in newer releases.
        scatter_map = getattr(px, "scatter_map", None)
        if scatter_map is not None:
            fig = scatter_map(
                group_df,
                map_style="open-street-map",
                **map_options,
            )
        else:
            fig = px.scatter_mapbox(
                group_df,
                mapbox_style="open-street-map",
                **map_options,
            )
        fig.update_layout(
            margin=dict(l=0, r=0, t=0, b=0),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        return plot_response(fig)
    except Exception:
        logger.exception("Geographic analytics failed")
        return api_error("Geographic analytics failed; see server logs", 500)


@app.route("/api/analytics/scatter")
def analytics_scatter():
    property_type = request.args.get("property_type", "flat")
    if property_type not in {"flat", "house"}:
        return api_error("property_type must be 'flat' or 'house'", 400)

    property_df = analytics_df[
        analytics_df["property_type"].astype(str).str.casefold()
        == property_type
    ].dropna(subset=["built_up_area", "price", "bedRoom"])

    fig = px.scatter(
        property_df,
        x="built_up_area",
        y="price",
        color="bedRoom",
        title=f"Area Vs Price ({property_type})",
        render_mode="svg",
    )
    return plot_response(apply_transparent_bg(fig))


@app.route("/api/analytics/pie")
def analytics_pie():
    sector = request.args.get("sector", "overall")
    if sector == "overall":
        pie_df = analytics_df.dropna(subset=["bedRoom"])
    else:
        available_sectors = set(analytics_df["sector"].dropna().astype(str))
        if sector not in available_sectors:
            return api_error(f"Unknown sector: {sector}", 400)
        pie_df = analytics_df[
            analytics_df["sector"].astype(str) == sector
        ].dropna(subset=["bedRoom"])

    fig = px.pie(
        pie_df,
        names="bedRoom",
        title=f"BHK Distribution ({sector})",
    )
    return plot_response(apply_transparent_bg(fig))


@app.route("/api/analytics/boxplot")
def analytics_boxplot():
    bhk_price_df = analytics_df[
        analytics_df["bedRoom"] <= 4
    ].dropna(subset=["bedRoom", "price"])
    fig = px.box(
        bhk_price_df,
        x="bedRoom",
        y="price",
        title="BHK Price Range",
    )
    return plot_response(apply_transparent_bg(fig))


@lru_cache(maxsize=1)
def get_feature_text() -> str:
    return build_wordcloud_text()


@app.route("/api/analytics/wordcloud")
def analytics_wordcloud():
    try:
        feature_text = get_feature_text()
        if not feature_text.strip():
            return api_error("No property feature text is available", 404)

        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color=CARD_BG_COLOR,
            colormap="Pastel1",
            stopwords={"s"},
            min_font_size=10,
        ).generate(feature_text)

        image_buffer = io.BytesIO()
        wordcloud.to_image().save(image_buffer, format="PNG")
        encoded = base64.b64encode(image_buffer.getvalue()).decode("utf-8")
        return jsonify({"image": encoded})
    except Exception:
        logger.exception("Word-cloud generation failed")
        return api_error("Word-cloud generation failed; see server logs", 500)


@app.route("/api/analytics/distplot")
def analytics_distplot():
    house_prices = analytics_df.loc[
        analytics_df["property_type"] == "house",
        "price",
    ].dropna()
    flat_prices = analytics_df.loc[
        analytics_df["property_type"] == "flat",
        "price",
    ].dropna()
    if house_prices.empty and flat_prices.empty:
        return api_error("No price data is available for the distribution", 404)

    fig, axis = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor(CARD_BG_COLOR)
    axis.set_facecolor(CARD_BG_COLOR)

    if not house_prices.empty:
        sns.histplot(
            house_prices,
            kde=len(house_prices) > 1,
            stat="density",
            label="House",
            ax=axis,
            element="step",
            color="#818cf8",
        )
    if not flat_prices.empty:
        sns.histplot(
            flat_prices,
            kde=len(flat_prices) > 1,
            stat="density",
            label="Flat",
            ax=axis,
            element="step",
            color="#34d399",
        )

    axis.legend()
    axis.set_xlabel("Price (Cr)")
    axis.set_ylabel("Density")
    axis.set_title("Property Type Price Distribution")

    image_buffer = io.BytesIO()
    fig.savefig(
        image_buffer,
        format="png",
        bbox_inches="tight",
        facecolor=fig.get_facecolor(),
    )
    plt.close(fig)
    image_buffer.seek(0)
    encoded = base64.b64encode(image_buffer.getvalue()).decode("utf-8")
    return jsonify({"image": encoded})


# ---------------- Recommender APIs ----------------

@app.route("/api/recommend/search", methods=["POST"])
def search_location():
    data = get_json_object()
    if data is None:
        return api_error("Request body must be a JSON object", 400)

    location = data.get("location")
    if not isinstance(location, str) or location not in location_df.columns:
        return api_error("Select a valid location", 400)

    try:
        radius_km = float(data.get("radius"))
        if not math.isfinite(radius_km) or radius_km <= 0:
            raise ValueError("radius must be a finite number greater than zero")
    except (TypeError, ValueError) as error:
        return api_error(str(error), 400)

    distances = location_df[location].dropna()
    results = distances[distances < radius_km * 1000].sort_values()
    response_results = [
        {
            "name": str(property_name),
            "distance": round(float(distance) / 1000, 1),
        }
        for property_name, distance in results.items()
    ]
    return jsonify({"status": "success", "results": response_results})


@app.route("/api/recommend/apartments", methods=["POST"])
def recommend_apartments():
    data = get_json_object()
    if data is None:
        return api_error("Request body must be a JSON object", 400)

    property_name = data.get("apartment")
    if not isinstance(property_name, str) or not property_name.strip():
        return api_error("Select an apartment to get recommendations", 400)

    try:
        model = load_recommendation_model()
    except (FileNotFoundError, RuntimeError) as error:
        return api_error(str(error), 503)
    except Exception:
        logger.exception("Could not load recommendation model")
        return api_error(
            "Could not load recommendation model; see server logs",
            500,
        )

    property_names = model["property_names"]
    query = property_name.strip().casefold()
    matching_indices = [
        index
        for index, name in enumerate(property_names)
        if str(name).strip().casefold() == query
    ]
    if not matching_indices:
        return api_error(f"Apartment not found: {property_name}", 404)

    query_index = matching_indices[0]
    similarity_scores = model["combined_similarity"][query_index]
    ranked_indices = np.argsort(similarity_scores)[::-1]
    ranked_indices = [
        int(index) for index in ranked_indices if index != query_index
    ][:5]

    results = [
        {
            "property": str(property_names[index]),
            "score": round(float(similarity_scores[index]), 4),
        }
        for index in ranked_indices
    ]
    return jsonify({"status": "success", "results": results})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    debug = os.environ.get("FLASK_DEBUG", "").casefold() in {"1", "true", "yes"}
    app.run(host="127.0.0.1", debug=debug, port=port)
