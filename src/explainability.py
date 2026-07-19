import os
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import config

def generate_shap_analysis(model, X_test, preprocessor):

    print('\nGenerating SHAP Explainability Analysis...')

    try:
        # Sample Test Data
        sample_size = min(1000, X_test.shape[0])

        np.random.seed(42)

        indices = np.random.choice(X_test.shape[0], sample_size, replace=False)

        X_sample = X_test[indices]

        if hasattr(X_sample, 'toarray'):
            X_sample = X_sample.toarray()

        feature_names = preprocessor.get_feature_names_out()

        X_sample = pd.DataFrame(X_sample, columns=feature_names)

        # Create SHAP Explainer
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)

        # Compatibility with different SHAP versions
        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        elif hasattr(shap_values, 'values'):
            shap_values = shap_values.values

            if shap_values.ndim == 3:
                shap_values = shap_values[:, :, 1]

        # SHAP Summary Plot
        plt.figure(figsize=(10, 8))

        shap.summary_plot(shap_values, X_sample, show=False, max_display=10)
        plt.tight_layout()

        plt.savefig(config.SHAP_SUMMARY_PLOT, dpi=300, bbox_inches='tight' )
        plt.close()

        # SHAP Feature Importance Plot
        plt.figure(figsize=(10, 8))

        shap.summary_plot(shap_values, X_sample, plot_type='bar', show=False, max_display=10)
        plt.tight_layout()

        plt.savefig(config.SHAP_FEATURE_IMPORTANCE, dpi=300, bbox_inches='tight')
        plt.close()

        # Feature Importance Table
        importance = pd.DataFrame({
            'Feature': feature_names,
            'Mean |SHAP|': np.abs(
                shap_values
            ).mean(axis=0)
        })

        importance = importance.sort_values(
            by='Mean |SHAP|',
            ascending=False
        )

        importance.to_csv(
            config.SHAP_FEATURE_TABLE,
            index=False
        )

        # Text Report
        top10 = importance.head(10)

        with open(config.SHAP_REPORT, 'w') as file:
            file.write('SHAP Explainability Report\n')
            file.write('=' * 60)
            file.write('\n\n')
            file.write(
                'Top 10 Most Important Features\n\n'
            )
            file.write(
                top10.to_string(index=False)
            )
        print('SHAP explainability completed successfully.')

    except Exception as e:
        print('\nSHAP Analysis Failed')
        print(e)