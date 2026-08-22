import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def plot_custom_confusion_matrix(y_true, y_pred, class_names=None, save_path=None, title="Confusion Matrix"):
    """
    Plots a highly readable confusion matrix using Seaborn.
    """
    if class_names is None:
        class_names = [f"Class {c}" for c in np.unique(y_true)]
        
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    
    plt.title(title)
    plt.ylabel('Actual Class')
    plt.xlabel('Predicted Class')
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
        
    plt.show()

def plot_feature_importance(model, feature_names, save_path=None, title="Feature Importance"):
    """
    Plots a horizontal bar chart of feature importances for tree-based models (RF, LightGBM).
    """
    # Extract importances based on model type
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    else:
        print("Model does not have 'feature_importances_' attribute.")
        return

    # Create a DataFrame for easy sorting and plotting
    df_imp = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    })
    
    # Sort features by importance
    df_imp = df_imp.sort_values(by='Importance', ascending=True)
    
    plt.figure(figsize=(10, 8))
    plt.barh(df_imp['Feature'], df_imp['Importance'], color='teal')
    plt.title(title)
    plt.xlabel("Importance Score")
    plt.ylabel("Features")
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Feature importance plot saved to {save_path}")
        
    plt.show()
