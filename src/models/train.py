import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
import pickle
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GroupKFold
from src.visualization.learning_curves import plot_sklearn_learning_curve
from src.visualization.evaluation_plots import plot_custom_confusion_matrix, plot_feature_importance

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
        
        # Save confusion matrix for the last fold to visualize typical validation errors
        if fold == 2:
            print("\nGenerating Confusion Matrix for Fold 3...")
            plot_custom_confusion_matrix(
                y_val, preds, 
                class_names=["Solar/Built", "Vegetation", "Bare Soil/Scrub", "Water"],
                save_path="docs/figures/rf_confusion_matrix.png",
                title="Random Forest Validation Confusion Matrix"
            )

    # Fit final model on full dataset
    final_rf = RandomForestClassifier(n_estimators=200, random_state=42)
    final_rf.fit(X, y)
    
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "rf_lulc_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump({'model': final_rf, 'feature_cols': feature_cols}, f)
        
    print(f"\nFinal model saved successfully to: {model_path}")

    # Generate and save Learning Curve
    print("\nGenerating Learning Curve...")
    plot_sklearn_learning_curve(
        estimator=RandomForestClassifier(n_estimators=100, random_state=42),
        X=X,
        y=y,
        cv=3, # Simplified CV for the plot
        save_path="docs/figures/rf_learning_curve.png"
    )

    # Generate and save Feature Importance Plot
    print("\nGenerating Feature Importance Plot...")
    plot_feature_importance(
        final_rf, 
        feature_cols, 
        save_path="docs/figures/rf_feature_importance.png",
        title="Random Forest Feature Importance"
    )

if __name__ == "__main__":
    train_and_evaluate()