"""Train and evaluate a linear regression model with categorical preprocessing."""
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "houses.csv"
FEATURES = ["area_m2", "bedrooms", "location"]
TARGET = "price_billion_vnd"
LOCATIONS = ("City Center", "Inner Suburb", "Outer Suburb")


def train_model(data_path=DATA_PATH):
    data = pd.read_csv(data_path)
    X, y = data[FEATURES], data[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    preprocessing = ColumnTransformer([
        ("location", OneHotEncoder(drop="first", handle_unknown="ignore"), ["location"]),
        ("numeric", "passthrough", ["area_m2", "bedrooms"]),
    ])
    model = Pipeline([
        ("preprocessing", preprocessing),
        ("regression", LinearRegression()),
    ])
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    metrics = {
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "mae": float(mean_absolute_error(y_test, predictions)),
        "rmse": float(mean_squared_error(y_test, predictions) ** 0.5),
        "r2": float(r2_score(y_test, predictions)),
    }
    return model, metrics, y_test.to_numpy(), predictions


def predict_price(model, area, bedrooms, location):
    if not 40 <= area <= 250:
        raise ValueError("Area must be between 40 and 250 m².")
    if bedrooms not in range(1, 7):
        raise ValueError("Bedrooms must be an integer between 1 and 6.")
    if location not in LOCATIONS:
        raise ValueError("Choose a supported location.")
    row = pd.DataFrame([[area, bedrooms, location]], columns=FEATURES)
    return float(model.predict(row)[0])
