import os
import pickle
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GroupKFold

def train_and_evaluate(feature_csv="data/processed/train_features.csv", model_dir="models"):
    df = pd.read_csv(feature_csv)
    
    # Feature columns exclude metadata
    meta_cols = ['class_id', 'class_name', 'longitude', 'latitude']
    feature_cols = [c for c in df.columns if c not in meta_cols]
    
    X = df[feature_cols].values
    y = df['class_id'].values
    
    # Create spatial blocks for Spatial Group K-Fold Cross Validation
    # Group coordinates into coarse spatial blocks to avoid spatial data leakage
    df['spatial_block'] = (df['longitude'].round(2).astype(str) + "_" + df['latitude'].round(2).astype(str))
    groups = df['spatial_block'].values

    gkf = GroupKFold(n_splits=3)
    
    print("\n--- Running Spatial Group K-Fold Cross-Validation ---")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=groups)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        rf_model.fit(X_train, y_train)
        preds = rf_model.predict(X_val)
        
        print(f"\nFold {fold + 1} Classification Report:")
        print(classification_report(y_val, preds, zero_division=0))

    # Fit final model on full dataset
    final_rf = RandomForestClassifier(n_estimators=200, random_state=42)
    final_rf.fit(X, y)
    
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "rf_lulc_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump({'model': final_rf, 'feature_cols': feature_cols}, f)
        
    print(f"\nFinal model saved successfully to: {model_path}")

if __name__ == "__main__":
    train_and_evaluate()