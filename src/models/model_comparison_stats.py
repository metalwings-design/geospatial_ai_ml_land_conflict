import pickle
import numpy as np
import pandas as pd
from scipy.stats import chi2
from sklearn.metrics import cohen_kappa_score, classification_report, accuracy_score, f1_score


def load_data_and_models(
    feature_csv="data/processed/train_features.csv",
    rf_path="models/rf_lulc_model.pkl",
    lgbm_path="models/lgbm_lulc_model.pkl"
):
    df = pd.read_csv(feature_csv)
    meta_cols = ['class_id', 'class_name', 'longitude', 'latitude']
    feature_cols = [c for c in df.columns if c not in meta_cols]
    
    X = df[feature_cols].values
    y = df['class_id'].values

    with open(rf_path, "rb") as f:
        rf_model = pickle.load(f)['model']

    with open(lgbm_path, "rb") as f:
        lgbm_model = pickle.load(f)['model']

    return X, y, feature_cols, rf_model, lgbm_model


def compute_statistical_tests():
    X, y, feature_cols, rf_model, lgbm_model = load_data_and_models()

    # Generate predictions
    rf_preds = rf_model.predict(X)
    lgbm_preds = lgbm_model.predict(X)

    # 1. Cohen's Kappa Score
    kappa_rf = cohen_kappa_score(y, rf_preds)
    kappa_lgbm = cohen_kappa_score(y, lgbm_preds)

    print("=======================================================")
    print("         INTER-MODEL ACCURACY & KAPPA METRICS          ")
    print("=======================================================")
    print(f"Random Forest Accuracy: {accuracy_score(y, rf_preds):.4f} | Kappa: {kappa_rf:.4f}")
    print(f"LightGBM Accuracy:      {accuracy_score(y, lgbm_preds):.4f} | Kappa: {kappa_lgbm:.4f}")

    # 2. McNemar's Test
    # Matrix construction:
    # e00: both correct, e01: RF correct & LGBM wrong, e10: RF wrong & LGBM correct, e11: both wrong
    rf_correct = (rf_preds == y)
    lgbm_correct = (lgbm_preds == y)

    e01 = np.sum(rf_correct & ~lgbm_correct)
    e10 = np.sum(~rf_correct & lgbm_correct)

    # McNemar's test statistic with continuity correction
    if (e01 + e10) > 0:
        stat = (abs(e01 - e10) - 1)**2 / (e01 + e10)
        p_val = chi2.sf(stat, df=1)
    else:
        stat, p_val = 0.0, 1.0

    print("\n=======================================================")
    print("       MCNEMAR'S TEST FOR MODEL EQUIVALENCE            ")
    print("=======================================================")
    print(f"RF Correct / LGBM Wrong (b): {e01}")
    print(f"RF Wrong / LGBM Correct (c): {e10}")
    print(f"Chi-Squared Statistic:       {stat:.4f}")
    print(f"p-value:                      {p_val:.4e}")
    if p_val > 0.05:
        print("Interpretation: No statistically significant difference between RF and LightGBM predictions (p > 0.05).")
    else:
        print("Interpretation: Statistically significant difference between RF and LightGBM predictions (p <= 0.05).")

    # 3. Feature Importance Comparison
    rf_importances = rf_model.feature_importances_
    lgbm_importances = lgbm_model.feature_importances_ / lgbm_model.feature_importances_.sum()

    df_imp = pd.DataFrame({
        'Feature': feature_cols,
        'RF_Importance': rf_importances,
        'LGBM_Importance': lgbm_importances
    }).sort_values(by='LGBM_Importance', ascending=False)

    print("\n=======================================================")
    print("           FEATURE IMPORTANCE COMPARISON               ")
    print("=======================================================")
    print(df_imp.to_string(index=False))

    df_imp.to_csv("data/processed/feature_importances_comparison.csv", index=False)
    print("\nFeature importance comparisons saved to: data/processed/feature_importances_comparison.csv")


if __name__ == "__main__":
    compute_statistical_tests()