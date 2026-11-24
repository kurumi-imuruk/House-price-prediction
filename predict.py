"""Optional command-line interface for a single house prediction."""
import argparse

from model import LOCATIONS, predict_price, train_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--area", type=float, required=True, help="Area in m² (40–250)")
    parser.add_argument("--bedrooms", type=int, required=True, help="Bedrooms (1–6)")
    parser.add_argument("--location", choices=LOCATIONS, required=True)
    args = parser.parse_args()
    model, _, _, _ = train_model()
    try:
        price = predict_price(model, args.area, args.bedrooms, args.location)
    except ValueError as error:
        parser.error(str(error))
    print(f"Estimated price: {price:.2f} billion VND (synthetic data)")


if __name__ == "__main__":
    main()
