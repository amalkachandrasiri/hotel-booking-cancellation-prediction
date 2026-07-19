import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import config

sns.set_theme(style='whitegrid')

# overview 
def dataset_overview(df):
    '''
    Display basic dataset information.
    '''

    print('=' * 60)
    print('DATASET OVERVIEW')
    print('=' * 60)

    print(f'\nDataset Shape : {df.shape}')

    print('\nData Types')
    print(df.dtypes)

    print('\nMissing Values')
    print(df.isnull().sum())

    print('\nDuplicate Records')
    print(df.duplicated().sum())

    print('\nSummary Statistics')
    print(df.describe(include='all'))

# missing values 
def missing_value_analysis(df):

    missing = (df.isnull().sum().sort_values(ascending=False))
    missing = missing[missing > 0]

    missing_percent = (missing / len(df) * 100)

    missing_df = pd.DataFrame({
        'Missing Values': missing,
        'Percentage': missing_percent
    })

    print('\nMissing Value Summary')
    print(missing_df)       

# duplicate values
def duplicate_analysis(df):

    duplicates = df.duplicated().sum()
    duplicate_percentage = (duplicates / len(df)) * 100

    print(f'Duplicate Records : {duplicates}')
    print(f'Duplicate Percentage : {duplicate_percentage:.2f}%')

# target variable analysis
def target_variable_analysis(df):
    '''
    Analyze the distribution of the target variable.
    '''

    print('\n' + '=' * 60)
    print('TARGET VARIABLE ANALYSIS')
    print('=' * 60)

    target_counts = df['is_canceled'].value_counts().sort_index()
    target_percentage = round(
        df['is_canceled'].value_counts(normalize=True) * 100, 2
    )

    summary = pd.DataFrame({
        'Count': target_counts,
        'Percentage (%)': target_percentage
    })

    print(summary)

    plt.figure(figsize=(6, 5))

    ax = sns.countplot(data = df,  x = 'is_canceled', palette = 'Set2')

    plt.title('Booking Cancellation Distribution', fontsize=14)
    plt.xlabel('Booking Status')
    plt.ylabel('Number of Bookings')

    ax.set_xticklabels(['Not Cancelled', 'Cancelled'])

    for p in ax.patches:
        ax.annotate(
            f'{int(p.get_height())}',
            (p.get_x() + p.get_width() / 2.,
             p.get_height()),
            ha='center',
            va='bottom',
            fontsize=10
        )

    plt.tight_layout()

    plt.savefig(config.TARGET_ANALYSIS, dpi=300, bbox_inches = 'tight')
    plt.show()

    return summary

# correlation analysis
def correlation_analysis(df):

    print('\n' + '=' * 60)
    print('CORRELATION ANALYSIS')
    print('=' * 60)

    # Numerical features only
    numerical_df = df.select_dtypes(include=['int64', 'float64'])

    correlation_matrix = numerical_df.corr()

    plt.figure(figsize=(14,10))

    sns.heatmap(
        correlation_matrix,
        cmap='coolwarm',
        center=0,
        linewidths=0.5,
        annot=False
    )

    plt.title('Correlation Heatmap')
    plt.tight_layout()
    plt.savefig(config.CORRELATION_ANALYSIS, dpi=300, bbox_inches = 'tight')
    plt.show()

    print('\nTop Correlated Features')

    corr_pairs = (
        correlation_matrix
        .abs()
        .unstack()
        .sort_values(ascending=False)
    )

    corr_pairs = corr_pairs[corr_pairs < 1]
    print(corr_pairs.head(15))

    return correlation_matrix

# outlier analysis
def outlier_analysis(df):

    print('\n' + '=' * 60)
    print('OUTLIER ANALYSIS')
    print('=' * 60)

    features = [
        'lead_time',
        'adr',
        'stays_in_week_nights',
        'stays_in_weekend_nights',
        'total_of_special_requests'
    ]

    for feature in features:

        Q1 = df[feature].quantile(0.25)
        Q3 = df[feature].quantile(0.75)

        IQR = Q3 - Q1

        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        outliers = df[
            (df[feature] < lower) |
            (df[feature] > upper)
        ]


        print(f'{feature}  - Outliers : {len(outliers)} - Percentage : {(len(outliers)/len(df))*100:.2f}%')
        print('-'*40)

# ---------------------------------------------
# ----------- Bussiness Insights -------------- 
# ---------------------------------------------

def categorical_business_insights(df):

    print('\n' + '=' * 60)
    print('CATEGORICAL BUSINESS INSIGHTS')
    print('=' * 60)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    categorical_features = [
        ('hotel', 'Hotel Type'),
        ('deposit_type', 'Deposit Type'),
        ('market_segment', 'Market Segment')
    ]

    summaries = {}

    for ax, (feature, title) in zip(axes, categorical_features):
        summary = (
            pd.crosstab(
                df[feature],
                df['is_canceled'],
                normalize='index'
            ) * 100
        ).round(2)

        summaries[feature] = summary

        print(f'\n{title}')
        print(summary)

        cancellation_rate = (
            df.groupby(feature)['is_canceled']
              .mean()
              .mul(100)
              .sort_values(ascending=False)
        )

        sns.barplot(
            x=cancellation_rate.index,
            y=cancellation_rate.values,
            ax=ax,
            palette='Set2'
        )

        ax.set_title(title)
        ax.set_xlabel('')
        ax.set_ylabel('Cancellation Rate (%)')
        ax.tick_params(axis='x', rotation=30)

    plt.tight_layout()
    plt.savefig(config.CAT_BUS_INSIGHTS, dpi = 300,  bbox_inches = 'tight')
    plt.show()

    return summaries

def numerical_business_insights(df):

    print('\n' + '=' * 60)
    print('NUMERICAL BUSINESS INSIGHTS')
    print('=' * 60)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sns.boxplot(
        data=df,
        x='is_canceled',
        y='lead_time',
        ax=axes[0]
    )

    axes[0].set_title('Lead Time vs Cancellation')
    axes[0].set_xlabel('Cancelled')
    axes[0].set_ylabel('Lead Time (Days)')

    sns.boxplot(
        data=df,
        x='is_canceled',
        y='adr',
        ax=axes[1]
    )

    axes[1].set_title('ADR vs Cancellation')
    axes[1].set_xlabel('Cancelled')
    axes[1].set_ylabel('Average Daily Rate')

    plt.tight_layout()
    plt.savefig(config.NUM_BUS_INSIGHTS, dpi = 300,  bbox_inches = 'tight')
    plt.show()

# wrapper 
def run_eda(df):
    # dataset_overview(df)
    # missing_value_analysis(df)
    # duplicate_analysis(df)
    # target_variable_analysis(df)
    # correlation_analysis(df)
    # outlier_analysis(df)
    categorical_summary = categorical_business_insights(df)
    numerical_business_insights(df)