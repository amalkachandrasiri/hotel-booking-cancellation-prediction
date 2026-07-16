import config
import pandas as pd

def fetch_data():
    df = pd.read_csv(config.RAW_DATA_PATH)
    return df

def describe_data(df):
    print(f'dataset shape - {df.shape[0]} rows X {df.shape[1]} columns')
    print(df.info())
    print(df.describe())

raw_df = fetch_data()
describe_data(raw_df)



print(raw_df.head(3))