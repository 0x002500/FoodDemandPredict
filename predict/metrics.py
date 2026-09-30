"""Backtest metrics used by the reporting pipeline."""
from __future__ import annotations
import numpy as np

def calculate_metrics(actual: object, predicted: object) -> dict[str, float]:
    """Return point-forecast error metrics; MAPE omits zero actuals."""
    actual_array, predicted_array = np.asarray(actual, dtype=float), np.asarray(predicted, dtype=float)
    error = predicted_array - actual_array; nonzero = actual_array != 0
    mape = float(np.mean(np.abs(error[nonzero] / actual_array[nonzero])) * 100) if nonzero.any() else float("nan")
    denominator = np.abs(actual_array) + np.abs(predicted_array); valid = denominator != 0
    smape = float(np.mean(2 * np.abs(error[valid]) / denominator[valid]) * 100) if valid.any() else float("nan")
    return {"MAE": float(np.mean(np.abs(error))), "RMSE": float(np.sqrt(np.mean(error**2))), "MAPE": mape, "sMAPE": smape}
