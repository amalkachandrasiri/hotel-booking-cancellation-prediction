import logging
from pathlib import Path
from typing import Final

# =========================
# Configuration
# =========================

BASE_DIR = Path(__file__).resolve().parent.parent

# data paths 
RAW_DATA_PATH       = BASE_DIR / 'data' / 'raw_data' / 'hotel_bookings.csv'
PROCESSED_DATA_PATH = BASE_DIR / 'data' / 'processed_data' / 'hotel_bookings_processed_dataset.csv'

# output figures 
TARGET_ANALYSIS      = BASE_DIR / 'output' / 'figures' / 'target_variable_analysis.png'
CORRELATION_ANALYSIS = BASE_DIR / 'output' / 'figures' / 'correlation_analysis.png'
OUTLIER_ANALYSIS     = BASE_DIR / 'output' / 'figures' / 'outlier_analysis.png'

CAT_BUS_INSIGHTS         = BASE_DIR / 'output' / 'figures' / 'cat_bus_insights.png'
NUM_BUS_INSIGHTS         = BASE_DIR / 'output' / 'figures' / 'num_bus_insights.png'

CLASS_DISTRIBUTION       = BASE_DIR / 'output' / 'figures' / 'class_distribution.png' 

SHAP_SUMMARY_PLOT        = BASE_DIR / 'output' / 'figures' / 'shap_summary_plot.png'
SHAP_FEATURE_IMPORTANCE  = BASE_DIR / 'output' / 'figures' / 'shap_feature_importance.png' 

ROC_CURVE_COMPARISON     = BASE_DIR / 'output' / 'figures' / 'roc_curve_comparison.png' 

# output reports 
CLASSIFICATION_REPORT    = BASE_DIR / 'output' / 'results' / 'model_classification_report.txt'
MODEL_COMPARISON_REPORT  = BASE_DIR / 'output' / 'results' / 'model_comparison_report.csv'
SHAP_FEATURE_TABLE       = BASE_DIR / 'output' / 'results' / 'shap_feature_importance.csv' 

SHAP_REPORT              = BASE_DIR / 'output' / 'results' / 'shap_report.txt'

CR_BY_MARKET_SGMNT       = BASE_DIR / 'output' / 'figures' / 'cr_by_market_sgmnt.png'
LEADTIME_VS_CANCELLATION = BASE_DIR / 'output' / 'figures' / 'leadtime_vs_cancellation.png'
ADR_VS_CANCELLATION      = BASE_DIR / 'output' / 'figures' / 'adr_vs_cancellation.png'

# artifacts 
CATBOOST_MODEL           = BASE_DIR / 'artifacts' / 'catboost_model.cbm'
PREPROCESSOR             = BASE_DIR / 'artifacts' / 'preprocessor.pkl'