import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import config
import tensorflow as tf
from explainability import generate_shap_analysis

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve
)

# Global Storage
model_results = []
roc_data = {}
trained_models = {}

# Evaluate a Model
def evaluate_model(model, X_test, y_test, model_name, preprocessor = None):

    print('\n' + '=' * 70)
    print(model_name.upper())
    print('=' * 70)

    # Predictions
    if isinstance(model, tf.keras.Model):
        y_prob = model.predict(X_test, verbose=0).flatten()
        y_pred = (y_prob >= 0.5).astype(int)
    else:
        y_pred = model.predict(X_test)

        if hasattr(model, 'predict_proba'):
            y_prob = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, 'decision_function'):
            y_prob = model.decision_function(X_test)
        else:
            y_prob = None

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    roc_auc = np.nan

    if y_prob is not None:
        roc_auc = roc_auc_score(y_test, y_prob)

    print(f'Accuracy : {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall   : {recall:.4f}')
    print(f'F1 Score : {f1:.4f}')

    if not np.isnan(roc_auc):
        print(f'ROC AUC  : {roc_auc:.4f}')

    # Classification Report
    report = classification_report(y_test, y_pred)

    with open(config.CLASSIFICATION_REPORT, 'a') as file:
        file.write('\n')
        file.write('=' * 70 + '\n')
        file.write(f'{model_name.upper()}\n')
        file.write('=' * 70 + '\n\n')

        file.write(f'Accuracy : {accuracy:.4f}\n')
        file.write(f'Precision: {precision:.4f}\n')
        file.write(f'Recall   : {recall:.4f}\n')
        file.write(f'F1 Score : {f1:.4f}\n')

        if not np.isnan(roc_auc):
            file.write(f'ROC AUC  : {roc_auc:.4f}\n')

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)

        file.write('\nConfusion Matrix\n')
        file.write(str(cm))
        file.write('\n\n')

        # Classification Report

        file.write('Classification Report\n\n')
        file.write(report)
        file.write('\n\n')

    # Store Results
    model_results.append({
        'Model': model_name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'ROC AUC': roc_auc
    })

    # Save ROC Curve
    if y_prob is not None:

        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_data[model_name] = (fpr, tpr, roc_auc)

    trained_models[model_name] = model

    print('x_test_type - ', type(X_test))

    # Generate SHAP analysis for the best model - applies for CatBoost only as it was choosen as the best model 
    if model_name == 'CatBoost'and preprocessor is not None:
        generate_shap_analysis(model, X_test, preprocessor)

# Compare Models
def compare_models():

    comparison_df = pd.DataFrame(model_results)

    comparison_df = comparison_df.sort_values(by='F1 Score', ascending=False).reset_index(drop=True)
    comparison_df.insert(0, 'Rank', range(1, len(comparison_df) + 1))

    comparison_df = comparison_df[
        ['Rank', 'Model', 'F1 Score', 'Accuracy', 'Precision', 'Recall', 'ROC AUC']
    ]

    numeric_cols = ['F1 Score', 'Accuracy', 'Precision', 'Recall', 'ROC AUC']
    comparison_df[numeric_cols] = comparison_df[numeric_cols].round(4)

    print('\n' + '=' * 90)
    print('MODEL COMPARISON')
    print('=' * 90)
    print(comparison_df)

    # save the results 
    with open(config.CLASSIFICATION_REPORT, 'a') as file:
        file.write('\n')
        file.write('=' * 90 + '\n')
        file.write('OVERALL MODEL COMPARISON\n')
        file.write('=' * 90 + '\n\n')

        file.write(comparison_df.to_string(index=False))
        file.write('\n\n')

    return comparison_df

# Plot ROC Curves
def plot_roc_curves():

    plt.figure(figsize=(8,6))

    for model_name, values in roc_data.items():
        fpr, tpr, auc = values
        plt.plot(
            fpr,
            tpr,
            label=f'{model_name} (AUC={auc:.3f})'
        )
    plt.plot(
        [0,1],
        [0,1],
        linestyle='--',
        color='black'
    )

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve Comparison')
    plt.legend()
    plt.tight_layout()

    plt.savefig(config.ROC_CURVE_COMPARISON, dpi=300, bbox_inches='tight')
    plt.show()