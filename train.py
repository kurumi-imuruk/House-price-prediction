"""Train the model and export evaluation metrics and a scatter plot."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model import ROOT, train_model


def main():
    _, metrics, actual, predicted = train_model()
    reports = ROOT / "reports"
    reports.mkdir(exist_ok=True)
    (reports / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(actual, predicted, alpha=0.65, color="#2563eb", edgecolors="none")
    low, high = min(actual.min(), predicted.min()), max(actual.max(), predicted.max())
    ax.plot([low, high], [low, high], "--", color="#64748b", label="Perfect prediction")
    ax.set(xlabel="Actual price (billion VND)", ylabel="Predicted price (billion VND)",
           title="House price prediction | Linear regression", xlim=(low-0.3, high+0.3),
           ylim=(low-0.3, high+0.3))
    ax.grid(alpha=0.15)
    ax.legend(frameon=False)
    ax.text(0.03, 0.96, f"R² = {metrics['r2']:.3f} | MAE = {metrics['mae']:.3f}",
            transform=ax.transAxes, va="top")
    fig.tight_layout()
    fig.savefig(reports / "actual_vs_predicted.png", dpi=150)
    plt.close(fig)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
