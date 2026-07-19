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

# output paths
TARGET_ANALYSIS      = BASE_DIR / 'output' / 'figures' / 'target_variable_analysis.png'
CORRELATION_ANALYSIS = BASE_DIR / 'output' / 'figures' / 'correlation_analysis.png'
OUTLIER_ANALYSIS     = BASE_DIR / 'output' / 'figures' / 'outlier_analysis.png'

CAT_BUS_INSIGHTS         = BASE_DIR / 'output' / 'figures' / 'cat_bus_insights.png'
NUM_BUS_INSIGHTS         = BASE_DIR / 'output' / 'figures' / 'num_bus_insights.png'

CLASS_DISTRIBUTION       = BASE_DIR / 'output' / 'figures' / 'class_distribution.png'


CR_BY_MARKET_SGMNT       = BASE_DIR / 'output' / 'figures' / 'cr_by_market_sgmnt.png'
LEADTIME_VS_CANCELLATION = BASE_DIR / 'output' / 'figures' / 'leadtime_vs_cancellation.png'
ADR_VS_CANCELLATION      = BASE_DIR / 'output' / 'figures' / 'adr_vs_cancellation.png'