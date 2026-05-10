"""
Run inference on the held-out regression test set with the two regression models
(NN_Reg_SHAP_artifact, XGB_reg) and write one submission pair per model to
Electron_Project/Submission:

    Regression_RasmusReimer_<ModelName>.csv             (index, p_Truth_Energy)
    Regression_RasmusReimer_<ModelName>_VariableList.csv (one feature per line)
"""

import os
# PyTorch and XGBoost both ship their own libomp on macOS; loading both into
# the same process segfaults inside OpenMP. Allow the duplicate and serialise
# OMP before either library is imported.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
import torch


# --- Paths --------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]   # .../Electron_Project
DATA_DIR     = PROJECT_ROOT / "Data"
REG_DIR      = PROJECT_ROOT / "Regression"
NN_DIR       = REG_DIR / "NN_Reg" / "saved_models"
XGB_DIR      = REG_DIR / "XGB_Regression" / "saved_models"
SUBMIT_DIR   = PROJECT_ROOT / "Submission"

TEST_H5 = DATA_DIR / "AppML_InitialProject_test_regression.h5"

TARGET_COL = "p_Truth_Energy"
SUBMITTER  = "RasmusReimer"

# Make Modules/ importable so we can reuse the trained-architecture class.
sys.path.append(str(PROJECT_ROOT))
from Modules.models import ThreeLayerRegressor


# --- Solutions ----------------------------------------------------------------
# Each entry maps a submission name to (kind, model file).
# Both models are self-describing — the NN .pth bundles weights+scaler+features+
# hparams, the XGB joblib pickle exposes `feature_names_in_`. So no separate
# feature-list file is needed for either.

SOLUTIONS = [
    ("NN_Reg",  "nn",  NN_DIR  / "NN_Reg_SHAP_artifact.pth"),
    ("XGB_Reg", "xgb", XGB_DIR / "XGB_reg.joblib"),
]


# --- Helpers ------------------------------------------------------------------

# Sanity cap on log(E) before exp(). The training-set energies span roughly
# log(E) ∈ [0, 8] for E in GeV. A handful of held-out rows have feature
# values thousands of σ from the training mean (sentinel-like values), and
# the NN extrapolates to log-preds in the thousands — exp() then overflows
# to inf and corrupts the CSV. Clipping keeps the submission finite without
# silently hiding it: we also log how many rows we clipped.
LOG_E_CLIP = 25.0   # exp(25) ≈ 7.2e10 GeV — already nonphysical, but finite


def _exp_with_clip(log_pred: np.ndarray, tag: str) -> np.ndarray:
    extreme = int(np.sum(log_pred > LOG_E_CLIP))
    if extreme:
        print(f"  WARNING [{tag}]: {extreme} row(s) have log-pred > {LOG_E_CLIP}; "
              f"clipping before exp() to keep CSV finite.")
    return np.exp(np.clip(log_pred, -LOG_E_CLIP, LOG_E_CLIP))


def predict_nn(artifact_path: Path, test_df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Load the bundled artifact (weights + scaler + features + params) and predict."""
    artifact = torch.load(artifact_path, map_location="cpu", weights_only=False)
    features  = list(artifact["features"])
    scaler    = artifact["scaler"]
    params    = artifact["params"]
    log_target = bool(artifact.get("log_target", True))

    # Pass the DataFrame (not .values) so the scaler matches by feature name.
    X = scaler.transform(test_df[features]).astype(np.float32)

    model = ThreeLayerRegressor(
        input_size=len(features),
        first_layer_size=params["first_layer"],
        second_layer_size=params["second_layer"],
        third_layer_size=params["third_layer"],
        dropout=params.get("dropout", 0.0),
    )
    model.load_state_dict(artifact["state_dict"])
    model.eval()
    with torch.no_grad():
        pred = model(torch.from_numpy(X)).cpu().numpy()

    return (_exp_with_clip(pred, "NN") if log_target else pred), features


def predict_xgb(model_path: Path, test_df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """XGBRegressor was trained on log(E), so undo that here."""
    model = joblib.load(model_path)
    features = list(model.feature_names_in_)
    log_pred = model.predict(test_df[features])
    return _exp_with_clip(log_pred, "XGB"), features


def write_submission(name: str, index: pd.Index, energy: np.ndarray, features: list[str]) -> None:
    """Format required by SubmissionChecker: no headers in either file."""
    SUBMIT_DIR.mkdir(parents=True, exist_ok=True)
    base = f"Regression_{SUBMITTER}_{name}"

    pd.DataFrame({"index": index, "p_Truth_Energy": energy}) \
      .to_csv(SUBMIT_DIR / f"{base}.csv", index=False, header=False)

    pd.Series(sorted(features)) \
      .to_csv(SUBMIT_DIR / f"{base}_VariableList.csv", index=False, header=False)

    print(f"  wrote {base}.csv  and  {base}_VariableList.csv")


# --- Main ---------------------------------------------------------------------

def main() -> None:
    test_df = pd.read_hdf(TEST_H5)
    if TARGET_COL in test_df.columns:                 # held-out set should not have it
        test_df = test_df.drop(columns=[TARGET_COL])
    print(f"Loaded test set: {test_df.shape[0]} rows, {test_df.shape[1]} columns")

    for name, kind, model_path in SOLUTIONS:
        print(f"\n[{name}] model={model_path.name}")

        if kind == "nn":
            energy, features = predict_nn(model_path, test_df)
        elif kind == "xgb":
            energy, features = predict_xgb(model_path, test_df)
        else:
            raise ValueError(f"unknown model kind: {kind}")

        print(f"  predicted energy range: [{energy.min():.2f}, {energy.max():.2f}] GeV")
        write_submission(name, test_df.index, energy, features)


if __name__ == "__main__":
    main()
