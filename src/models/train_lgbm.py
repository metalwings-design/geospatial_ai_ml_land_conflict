import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
import pickle
import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GroupKFold

def train_lgbm(feature_csv="data/processed/train_features.csv", model_dir="models"):
    df = pd.read_csv(feature_csv)
    
    meta_cols = ['class_id', 'class_name', 'longitude', 'latitude']
    feature_cols = [c for c in df.columns if c not in meta_cols]
    
    X = df[feature_cols].values
    y = df['class_id'].values
    
    # Spatial grouping to prevent data leakage during cross-validation
    df['spatial_block'] = (df['longitude'].round(2).astype(str) + "_" + df['latitude'].round(2).astype(str))
    groups = df['spatial_block'].values

    gkf = GroupKFold(n_splits=3)
    
    print("\n--- Running LightGBM Spatial Group K-Fold Cross-Validation ---")
    lgbm_model = LGBMClassifier(
        n_estimators=100, 
        learning_rate=0.05, 
        max_depth=6, 
        random_state=42,
        verbose=-1
    )
    
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=groups)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        lgbm_model.fit(X_train, y_train)
        preds = lgbm_model.predict(X_val)
        
        print(f"\nLightGBM Fold {fold + 1} Classification Report:")
        print(classification_report(y_val, preds, zero_division=0))

    # Fit final LightGBM model on full dataset
    final_lgbm = LGBMClassifier(
        n_estimators=150, 
        learning_rate=0.05, 
        max_depth=6, 
        random_state=42,
        verbose=-1
    )
    final_lgbm.fit(
        X, y,
        eval_set=[(X, y)],
    )
    
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "lgbm_lulc_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump({'model': final_lgbm, 'feature_cols': feature_cols}, f)
        
    print(f"\nLightGBM model saved successfully to: {model_path}")

    # Generate and save Loss Curve
    from src.visualization.learning_curves import plot_lightgbm_loss_curve
    if hasattr(final_lgbm, 'evals_result_'):
        print("\nGenerating Loss Curve...")
        plot_lightgbm_loss_curve(final_lgbm.evals_result_, save_path="docs/figures/lgbm_loss_curve.png")

if __name__ == "__main__":
    train_lgbm()