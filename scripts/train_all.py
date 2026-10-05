import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score

# Ensure scripts can import ml modules
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.generate_ml_dataset import generate_dataset
from ml.regime.train import train_regime_model
from ml.correction.train import train_correction_models
from ml.probability.train import train_probability_model
from ml.verification.metrics import (
    calculate_rmse, calculate_mae, calculate_contingency,
    calculate_csi, calculate_pod, calculate_far, calculate_ets, calculate_fss_proxy
)

def run_training_pipeline():
    print("Generating dataset...")
    generate_dataset(10000, "data/sample/training_data.csv")
    
    df = pd.read_csv("data/sample/training_data.csv")
    train_df, test_df = train_test_split(df, test_size=0.3, random_state=42)
    
    print("Training regime classifier...")
    regime_model = train_regime_model(train_df)
    
    print("Training correction models...")
    correction_models = train_correction_models(train_df)
    
    print("Training probability model...")
    prob_model = train_probability_model(train_df)
    
    print("Evaluating models on test set...")
    # Regime Eval
    X_test = test_df[[
        "gfs_rain_mm", "gfs_mslp", "gfs_t2m", "gfs_rh2m", "gfs_u10", "gfs_v10",
        "gfs_u850", "gfs_v850", "gfs_cape", "gfs_precipitable_water",
        "rain_last_6h", "rain_last_24h", "latitude", "longitude"
    ]]
    y_regime_true = test_df["regime"]
    y_regime_pred = regime_model.predict(X_test)
    
    regime_acc = accuracy_score(y_regime_true, y_regime_pred)
    regime_f1 = f1_score(y_regime_true, y_regime_pred, average="macro")
    
    # Heavy Rain Eval
    y_prob_true = test_df["heavy_rain"]
    y_prob_pred = prob_model.predict(X_test)
    y_prob_score = prob_model.predict_proba(X_test)[:, 1] if 1 in prob_model.classes_ else np.zeros(len(X_test))
    
    hr_auc = roc_auc_score(y_prob_true, y_prob_score) if len(np.unique(y_prob_true)) > 1 else 0.0
    hr_precision = precision_score(y_prob_true, y_prob_pred, zero_division=0)
    hr_recall = recall_score(y_prob_true, y_prob_pred, zero_division=0)
    
    # Correction Eval
    y_rain_true = test_df["actual_rain_mm"].values
    raw_rain = test_df["gfs_rain_mm"].values
    
    anvaya_corrected = []
    for i, row in test_df.iterrows():
        regime = row["regime"]
        if regime in correction_models:
            pred_log = correction_models[regime].predict(row[X_test.columns].to_frame().T)[0]
            corrected = max(0.0, np.expm1(pred_log))
        else:
            corrected = row["gfs_rain_mm"]
        anvaya_corrected.append(corrected)
        
    anvaya_corrected = np.array(anvaya_corrected)
    
    # Save generic metrics
    model_metrics = {
        "regime_accuracy": regime_acc,
        "regime_macro_f1": regime_f1,
        "heavy_rain_roc_auc": hr_auc,
        "heavy_rain_precision": hr_precision,
        "heavy_rain_recall": hr_recall
    }
    
    with open("data/sample/model_metrics.json", "w") as f:
        json.dump(model_metrics, f, indent=2)
        
    # Verification Metrics (Raw vs ANVAYA)
    print("Calculating verification metrics...")
    # Raw metrics
    raw_hits, raw_misses, raw_fa, raw_cn = calculate_contingency(y_rain_true, raw_rain)
    raw_csi = calculate_csi(raw_hits, raw_misses, raw_fa)
    
    # ANVAYA metrics
    hits, misses, fa, cn = calculate_contingency(y_rain_true, anvaya_corrected)
    
    rmse = calculate_rmse(y_rain_true, anvaya_corrected)
    ets = calculate_ets(hits, misses, fa, cn)
    csi = calculate_csi(hits, misses, fa)
    pod = calculate_pod(hits, misses)
    far = calculate_far(hits, fa)
    fss = calculate_fss_proxy(y_rain_true, anvaya_corrected)
    
    verification_metrics = {
        "metrics": {
            "rmse": float(rmse),
            "ets": float(ets),
            "csi": float(csi),
            "pod": float(pod),
            "far": float(far),
            "fss": float(fss)
        },
        "period": "Prototype validation on generated holdout data",
        "metadata": {
            "source": "demo",
            "sample_size": len(test_df),
            "raw_csi_baseline": float(raw_csi)
        }
    }
    
    with open("data/sample/verification_metrics.json", "w") as f:
        json.dump(verification_metrics, f, indent=2)
        
    print("Training and verification complete!")

if __name__ == "__main__":
    run_training_pipeline()
