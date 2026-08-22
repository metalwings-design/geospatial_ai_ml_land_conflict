import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve

def plot_sklearn_learning_curve(estimator, X, y, cv, title="Learning Curve", scoring='accuracy', train_sizes=np.linspace(0.1, 1.0, 5), save_path=None):
    """
    Generates and plots a learning curve for scikit-learn compatible models (like Random Forest).
    This proves whether the model is overfitting, underfitting, or well-fit.
    """
    plt.figure(figsize=(10, 6))
    plt.title(title)
    plt.xlabel("Training Examples")
    plt.ylabel(f"Score ({scoring})")

    # Compute learning curve data
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=-1, train_sizes=train_sizes, scoring=scoring
    )

    # Calculate means and standard deviations
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    # Plot the bands
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")

    # Plot the lines
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training Score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation Score")

    plt.legend(loc="best")
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Learning curve saved to {save_path}")
    
    plt.show()


def plot_lightgbm_loss_curve(evals_result, metric='multi_logloss', save_path=None):
    """
    Plots the training and validation loss curves from a LightGBM evals_result dictionary.
    Requires passing `evals_result` back from the `lgb.train()` function.
    """
    plt.figure(figsize=(10, 6))
    plt.title("LightGBM Training vs Validation Loss Curve")
    plt.xlabel("Boosting Rounds (Trees)")
    plt.ylabel(f"Loss ({metric})")
    
    # LightGBM keys depend on how eval_set was passed. It might be 'training' or 'valid_0'
    train_key = 'training' if 'training' in evals_result else list(evals_result.keys())[0]
    train_loss = evals_result[train_key][metric]
    
    plt.plot(train_loss, label='Training Loss', color='blue')
    
    # Check if a validation set was provided
    if len(evals_result) > 1:
        val_key = list(evals_result.keys())[1]
        val_loss = evals_result[val_key][metric]
        plt.plot(val_loss, label='Validation Loss', color='orange')
    
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc="best")
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"LightGBM loss curve saved to {save_path}")
        
    plt.show()


def plot_unet_loss_curve(train_losses, val_losses, save_path=None):
    """
    Plots the training and validation loss curves for a U-Net model across epochs.
    `train_losses` and `val_losses` should be lists of loss values recorded at each epoch.
    """
    epochs = range(1, len(train_losses) + 1)
    
    plt.figure(figsize=(10, 6))
    plt.title("U-Net Epoch-wise Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (e.g., Cross Entropy)")
    
    plt.plot(epochs, train_losses, 'b-o', label='Training Loss')
    plt.plot(epochs, val_losses, 'r-s', label='Validation Loss')
    
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc="best")
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"U-Net loss curve saved to {save_path}")
        
    plt.show()


if __name__ == "__main__":
    # --- Example Usage / Mock Data ---
    print("This script provides functions to plot learning and loss curves.")
    print("Import these functions into your training scripts (e.g., train.py, train_lgbm.py, train_unet.py).")
    
    # 1. Mocking U-Net Data Plotting Example
    mock_train_loss = [2.5, 1.8, 1.2, 0.9, 0.7, 0.55, 0.45, 0.4, 0.38, 0.35]
    mock_val_loss =   [2.4, 1.9, 1.4, 1.1, 0.9, 0.85, 0.82, 0.80, 0.81, 0.83] # Notice slight rise at end (overfitting starts)
    
    # plot_unet_loss_curve(mock_train_loss, mock_val_loss, save_path="../../docs/figures/mock_unet_loss.png")
