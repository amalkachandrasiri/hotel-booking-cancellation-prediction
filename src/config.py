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