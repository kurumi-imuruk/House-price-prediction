"""Create reproducible synthetic housing data for a regression demo."""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent


def generate_data(n=600, seed=42):
    rng = np.random.default_rng(seed)
    area = rng.integers(40, 251, n)
    bedrooms = np.clip(np.rint(area / 45 + rng.normal(0, 0.7, n)), 1, 6).astype(int)
    location = rng.choice(["City Center", "Inner Suburb", "Outer Suburb"], n)
    premiums = {"City Center": 2.5, "Inner Suburb": 1.2, "Outer Suburb": 0.0}
    # A simple, invented pricing rule plus random noise; not market data.
    price = 0.45 + 0.035 * area + 0.25 * bedrooms + np.array(
        [premiums[item] for item in location]
    ) + rng.normal(0, 0.45, n)
    return pd.DataFrame({
        "area_m2": area, "bedrooms": bedrooms, "location": location,
        "price_billion_vnd": np.round(price, 2),
    })


if __name__ == "__main__":
    path = ROOT / "data" / "houses.csv"
    path.parent.mkdir(exist_ok=True)
    generate_data().to_csv(path, index=False)
    print(f"Saved dataset: {path}")
