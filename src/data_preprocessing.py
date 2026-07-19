import config
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

def fetch_data(path):
    df = pd.read_csv(path)
    return df

def save_data(df, path):
    df.to_csv(path)    

def describe_data(df):
    print(f'dataset shape - {df.shape[0]} rows X {df.shape[1]} columns')
    print(df.info())
    print(df.describe())

def preprocess_data(df):

    data = df.copy()

    # Feature Selection

    columns_to_drop = ['company', 'reservation_status', 'reservation_status_date']
    data.drop(columns=columns_to_drop, inplace=True)

    print('Dropped Columns:')
    print(columns_to_drop)

    # handling Missing Value 

    data['children'] = data['children'].fillna(0)
    data['agent'] = data['agent'].fillna(0)
    data['country'] = data['country'].fillna(data['country'].mode()[0])

    print('\nRemaining Missing Values')
    print(data.isnull().sum()[data.isnull().sum() > 0])

    # Split Features and Target

    X = data.drop('is_canceled', axis=1)
    y = data['is_canceled']

    # Train/Test Split

    X_train, X_test, y_train, y_test = train_test_split(X,  y, test_size=0.20, random_state=42, stratify=y)

    print(f'\nTraining Set : {X_train.shape}')
    print(f'Testing Set  : {X_test.shape}')

    # Identify Feature Types

    categorical_features = X_train.select_dtypes(include=['object']).columns.tolist()
    numerical_features = X_train.select_dtypes(exclude=['object']).columns.tolist()

    print(f'\nCategorical Features : {len(categorical_features)}')
    print(f'Numerical Features   : {len(numerical_features)}')

    #Preprocessing Pipeline

    preprocessor = ColumnTransformer(
        transformers=[
            (
                'num',
                StandardScaler(),
                numerical_features
            ),
            (
                'cat',
                OneHotEncoder(
                    handle_unknown='ignore'
                ),
                categorical_features
            )
        ]
    )    

    #  Fit on Training Data
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    print('\nProcessed Training Shape :', X_train_processed.shape)
    print('Processed Testing Shape  :', X_test_processed.shape)

    print('\nPreprocessing Completed Successfully.')

    return (
        X_train_processed,
        X_test_processed,
        y_train,
        y_test,
        preprocessor
    )

def check_class_distribution(y, y_train, y_test):

    summary = pd.DataFrame({
        'Original (%)': (y.value_counts(normalize=True) * 100).round(2),
        'Training (%)': (y_train.value_counts(normalize=True) * 100).round(2),
        'Testing (%)': (y_test.value_counts(normalize=True) * 100).round(2)
    })
    summary.index = ['Not Cancelled', 'Cancelled']

    print(summary)
