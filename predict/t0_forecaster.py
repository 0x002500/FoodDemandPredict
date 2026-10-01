"""Validated wrapper around The Forecasting Company's t0-beta model."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import torch
from t0 import T0Forecaster
MODEL_ID = "theforecastingcompany/t0-beta"

@dataclass(frozen=True)
class ForecastResult:
    median: np.ndarray
    p10: np.ndarray
    p90: np.ndarray

class T0DemandPredictor:
    """Forecast one or many demand series with known-future covariates."""
    def __init__(self, model_id: str = MODEL_ID, device: str = "cpu") -> None:
        self.model_id, self.device, self._model = model_id, device, None
    @property
    def model(self) -> T0Forecaster:
        if self._model is None: self._model = T0Forecaster.from_pretrained(self.model_id).to(self.device).eval()
        return self._model
    def predict(self, history: np.ndarray, historical_covariates: np.ndarray, future_covariates: np.ndarray) -> ForecastResult:
        batch = self.predict_many(np.asarray(history, dtype=np.float32).reshape(1, -1), historical_covariates, future_covariates)
        return ForecastResult(median=batch.median[0], p10=batch.p10[0], p90=batch.p90[0])

    def predict_many(self, histories: np.ndarray, historical_covariates: np.ndarray, future_covariates: np.ndarray) -> ForecastResult:
        """Forecast multiple dish series in one t0-beta inference call.

        ``histories`` is ``[dish, history]``. Calendar covariates are shared
        across dishes, then repeated into t0's required ``[batch, feature,
        history + horizon]`` future-covariate tensor.
        """
        histories = np.asarray(histories, dtype=np.float32)
        historical_covariates, future_covariates = np.asarray(historical_covariates, dtype=np.float32), np.asarray(future_covariates, dtype=np.float32)
        if histories.ndim != 2 or histories.shape[0] == 0 or histories.shape[1] == 0: raise ValueError("histories 必须是二维数组 [序列, 时间]")
        if historical_covariates.ndim != 2 or future_covariates.ndim != 2: raise ValueError("协变量必须是二维数组 [时间, 特征]")
        if historical_covariates.shape[0] != histories.shape[1]: raise ValueError("历史协变量行数必须等于 history 长度")
        if historical_covariates.shape[1] != future_covariates.shape[1]: raise ValueError("历史与未来协变量的特征数必须一致")
        if future_covariates.shape[0] == 0: raise ValueError("未来协变量不能为空")
        if not (np.isfinite(histories).all() and np.isfinite(historical_covariates).all() and np.isfinite(future_covariates).all()): raise ValueError("目标或协变量含非有限数值")
        covariates = np.concatenate([historical_covariates, future_covariates], axis=0).T[None, :, :]
        covariates = np.repeat(covariates, histories.shape[0], axis=0)
        with torch.inference_mode():
            forecast = self.model.predict(torch.as_tensor(histories, device=self.device), horizon=future_covariates.shape[0], quantile_levels=[0.1, 0.5, 0.9], future_covariates=torch.as_tensor(covariates, device=self.device))
        quantiles = forecast.quantiles.detach().cpu().numpy()
        return ForecastResult(median=quantiles[:, :, 1], p10=quantiles[:, :, 0], p90=quantiles[:, :, 2])
