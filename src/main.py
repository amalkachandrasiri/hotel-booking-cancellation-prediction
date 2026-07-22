import config
import data_preprocessing as data_prep
from eda import run_eda
from evaluation import evaluate_model

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

from catboost import CatBoostClassifier

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Input
from tensorflow.keras.regularizers import l2

import joblib

import mlflow
import mlflow.catboost

from evaluation import (
    evaluate_model,
    compare_models,
    plot_roc_curves
)
from explainability import generate_shap_analysis
from explainability import validate_shap_features

# fetch raw data
df = data_prep.fetch_data(config.RAW_DATA_PATH)
# run_eda(df)

X_train, X_test, y_train, y_test, preprocessor = data_prep.preprocess_data(df)

data_prep.check_class_distribution(df['is_canceled'], y_train, y_test)

# store processed data 
data_prep.save_data(df, config.PROCESSED_DATA_PATH)

# -------------   Model Training ---------------------

# Logistic Regression
print('\nTraining Logistic Regression...')

lr_model = LogisticRegression(random_state = 42, max_iter = 1000)
lr_model.fit(X_train, y_train)

evaluate_model(lr_model, X_test, y_test, 'Logistic Regression', preprocessor)

# KNN
print('\nTraining KNN...')

knn_model = KNeighborsClassifier(n_neighbors=5, weights='uniform')
knn_model.fit(X_train, y_train)

evaluate_model(knn_model, X_test, y_test, 'KNN')

# RANDOM FOREST
print('\nTraining Random Forest...')

rf_model = RandomForestClassifier(n_estimators=300, max_depth = 20, min_samples_split = 5, random_state = 42, n_jobs = -1)
rf_model.fit(X_train, y_train)

evaluate_model(rf_model, X_test, y_test, 'Random Forest')


mlflow.set_tracking_uri(f'sqlite:///{config.MLFLOW_DB_PATH.as_posix()}')
mlflow.set_experiment('hotel-booking-cancellation')
print('\nMLflow tracking URI:', mlflow.get_tracking_uri())

# CATBOOST
print('\nTraining CatBoost...')

with mlflow.start_run(run_name='catboost-final-model'):
    cat_model = CatBoostClassifier(iterations=300, learning_rate=0.1, depth=6, random_seed=42, verbose=0)
    cat_model.fit(X_train, y_train)

    # Save local artifacts
    cat_model.save_model(config.CATBOOST_MODEL)
    joblib.dump(preprocessor, config.PREPROCESSOR)

    # Evaluate model
    results = evaluate_model(cat_model, X_test, y_test, 'CatBoost', preprocessor)

    # Log parameters
    mlflow.log_params({
        'model_type': 'CatBoostClassifier',
        'iterations': 300,
        'learning_rate': 0.1,
        'depth': 6,
        'random_seed': 42
    })

    # Log metrics
    mlflow.log_metrics({
        'accuracy': results['Accuracy'],
        'precision': results['Precision'],
        'recall': results['Recall'],
        'f1_score': results['F1 Score'],
        'roc_auc': results['ROC AUC']
    })

    # Log artifacts
    mlflow.log_artifact(
        str(config.CATBOOST_MODEL),
        artifact_path='artifacts'
    )

    mlflow.log_artifact(
        str(config.PREPROCESSOR),
        artifact_path='artifacts'
    )


# ANN
print('\nTraining ANN...')

ann_model = Sequential([
    Input(shape=(X_train.shape[1],)),
    Dense(128, activation='relu', kernel_regularizer=l2(0.001)),
    Dropout(0.3),
    Dense(64, activation='relu', kernel_regularizer=l2(0.001)),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

ann_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'] )
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
history = ann_model.fit(X_train, y_train, validation_split=0.2, epochs=50, batch_size=256, callbacks=[early_stop], verbose=1)

ann_model.summary()

evaluate_model(ann_model, X_test, y_test, 'ANN')

# FINAL COMPARISON
results = compare_models()
plot_roc_curves()

print('\nTraining Completed Successfully.')

# validate SHAP features
validate_shap_features(df)
