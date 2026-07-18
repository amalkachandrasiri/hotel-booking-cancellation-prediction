from data_preprocessing import fetch_data
from eda import run_eda

df = fetch_data()
run_eda(df)