from data_processing.data_cleanup import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pprint import pprint
import random
import sys
import os


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


plt.style.use('ggplot')
sns.set_theme(style="whitegrid")


def print_section(title):
    """Print a section header for better readability."""
    print("\n" + "="*80)
    print(f" {title} ".center(80, "="))
    print("="*80 + "\n")


def print_report(report, max_depth=2):
    """Print a nested dictionary report with controlled depth."""
    import json
    print(json.dumps(report, indent=2, default=str)
          [:1000] + "...\n(Report truncated)")
    print("\n")


def compare_dataframes(before_df, after_df, title="DataFrame Comparison"):
    """Compare before and after DataFrames."""
    print(f"\n--- {title} ---\n")

    print(f"Shape Before: {before_df.shape}, After: {after_df.shape}")
    print(f"Data Types Before: \n{before_df.dtypes.to_string()}\n")
    print(f"Data Types After: \n{after_df.dtypes.to_string()}\n")

    if before_df.shape != after_df.shape:
        print("Can't show direct comparison as shapes differ")
        return

    print("\nSample data comparison:")
    sample_rows = min(5, len(before_df))
    pd.set_option('display.max_columns', 10)
    print("\nBefore:")
    print(before_df.head(sample_rows))
    print("\nAfter:")
    print(after_df.head(sample_rows))


def plot_outliers(df, column, outliers_mask=None, title="Outlier Detection"):
    """Plot a boxplot showing outliers in a column."""
    plt.figure(figsize=(10, 6))
    if outliers_mask is not None:
        plt.scatter(
            x=df.index[outliers_mask],
            y=df.loc[outliers_mask, column],
            color='red',
            label='Outliers'
        )
    sns.boxplot(x=df[column])
    plt.title(f"{title} for {column}")
    plt.tight_layout()
    plt.show()


def visualize_categorical_distribution(df, column, title="Category Distribution"):
    """Visualize the distribution of values in a categorical column."""
    plt.figure(figsize=(10, 6))
    df[column].value_counts().plot(kind='bar')
    plt.title(f"{title} for {column}")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()


def create_test_data(rows=1000):
    """Create a sample dataset with various data quality issues."""
    np.random.seed(42)
    random.seed(42)

    df = pd.DataFrame()

    df['id'] = range(1, rows+1)

    df['income'] = np.random.normal(60000, 15000, rows)
    df.loc[random.sample(range(rows), 50), 'income'] = None
    df.loc[random.sample(range(rows), 20), 'income'] = df.loc[random.sample(
        range(rows), 20), 'income'] * 5

    df['age'] = np.random.randint(18, 70, rows)
    df.loc[random.sample(range(rows), 30), 'age'] = df.loc[random.sample(
        range(rows), 30), 'age'].astype(str)
    df.loc[random.sample(range(rows), 10), 'age'] = 'twenty five'
    df.loc[random.sample(range(rows), 10), 'age'] = 'thirty two'
    df.loc[random.sample(range(rows), 5), 'age'] = None

    df['score'] = np.random.normal(75, 15, rows)
    df.loc[random.sample(range(rows), 40), 'score'] = df.loc[random.sample(
        range(rows), 40), 'score'].astype(str) + '%'
    df.loc[random.sample(range(rows), 20), 'score'] = None

    dates = pd.date_range(start='2020-01-01', periods=rows, freq='D')
    df['signup_date'] = dates

    date_formats = [
        lambda d: d.strftime('%Y-%m-%d'),
        lambda d: d.strftime('%m/%d/%Y'),
        lambda d: d.strftime('%d-%b-%Y'),
        lambda d: d.strftime('%Y%m%d')
    ]

    df['last_login'] = [date_formats[i %
                                     len(date_formats)](d) for i, d in enumerate(dates)]
    df.loc[random.sample(range(rows), 40), 'last_login'] = None

    cities = ['New York', 'los angeles', 'CHICAGO',
              'Houston', 'Phoenix', 'san antonio', 'PHILADELPHIA']
    df['city'] = [random.choice(cities) for _ in range(rows)]

    comments = [
        "Great product, would buy again!",
        "   not bad   ",
        "FANTASTIC SERVICE!!!",
        "Just okay; needs improvement.",
        "good product - will recommend",
        "Bad experience, won't purchase again...",
        "N/A",
        "very GOOD, but expensive$$!"
    ]
    df['comment'] = [random.choice(comments) for _ in range(rows)]
    df.loc[random.sample(range(rows), 60), 'comment'] = None

    categories = ['Category A', 'CategoryA', 'category a', 'Cat. A', 'Cat A',
                  'Category B', 'CategoryB', 'category b', 'cat B',
                  'Category C', 'CategoryC', 'category c']
    df['category'] = [random.choice(categories) for _ in range(rows)]

    statuses = ['active', 'inactive', 'pending', 'suspended']
    df['status'] = [random.choice(statuses) for _ in range(rows)]

    ratings = ['1', '2', '3', '4', '5', 'one', 'two', 'three', 'four', 'five']
    df['rating'] = [random.choice(ratings) for _ in range(rows)]

    df['price'] = ['$' + str(round(random.uniform(10, 1000), 2))
                   for _ in range(rows)]
    df.loc[random.sample(range(rows), 50), 'price'] = '£' + \
        df.loc[random.sample(range(rows), 50), 'price'].str[1:]
    df.loc[random.sample(range(rows), 50), 'price'] = '€' + \
        df.loc[random.sample(range(rows), 50), 'price'].str[1:]

    duplicate_rows = df.sample(100).copy()
    df = pd.concat([df, duplicate_rows], ignore_index=True)

    df['Customer ID'] = range(1, len(df)+1)
    df['first_name'] = ['User' + str(i) for i in range(len(df))]
    df['LastName'] = ['Surname' + str(i) for i in range(len(df))]

    return df


def main():
    print_section("Creating Test Data")
    df = create_test_data()
    print(f"Created test data with shape: {df.shape}")
    print("Sample data:")
    print(df.head())
    print("\nData types:")
    print(df.dtypes)
    print("\nMissing values per column:")
    print(df.isna().sum())

    original_df = df.copy()

    print_section("Testing handle_null_values")

    clean_df_mean, report_mean = handle_null_values(df, method='mean')
    print("Report for null handling with 'mean' method:")
    print_report(report_mean)
    compare_dataframes(df, clean_df_mean, "Null Handling (Mean)")

    clean_df_zero, report_zero = handle_null_values(df, method='zero')
    print("Report for null handling with 'zero' method:")
    print_report(report_zero)

    print_section("Testing fix_numeric_datatypes")
    clean_df_numeric, report_numeric = fix_numeric_datatypes(df)
    print("Report for numeric data type fixes:")
    print_report(report_numeric)
    compare_dataframes(df, clean_df_numeric, "Numeric Data Type Fixes")

    print_section("Testing detect_and_fix_date_columns")
    clean_df_dates, report_dates = detect_and_fix_date_columns(df)
    print("Report for date column fixes:")
    print_report(report_dates)
    compare_dataframes(df, clean_df_dates, "Date Column Fixes")

    print_section("Testing handle_outliers")

    clean_df_outliers_report, report_outliers = handle_outliers(
        df, method='iqr', action='report')
    print("Report for outlier detection (report only):")
    print_report(report_outliers)

    income_outliers = report_outliers['columns'].get('income', {})
    if 'outlier_count' in income_outliers and income_outliers['outlier_count'] > 0:
        outlier_mask = (df['income'] < income_outliers['bounds']['lower']) | (
            df['income'] > income_outliers['bounds']['upper'])
        plot_outliers(df, 'income', outlier_mask, "Income Outliers")

    clean_df_outliers_remove, report_outliers_remove = handle_outliers(
        df, method='iqr', action='remove')
    print("Report for outlier handling (remove):")
    print_report(report_outliers_remove)
    compare_dataframes(df, clean_df_outliers_remove, "Outlier Removal")

    print_section("Testing clean_text_columns")
    clean_df_text, report_text = clean_text_columns(
        df, columns=['comment', 'city'], lowercase=True, strip_spaces=True)
    print("Report for text cleaning:")
    print_report(report_text)
    compare_dataframes(df[['comment', 'city']], clean_df_text[[
                       'comment', 'city']], "Text Cleaning")

    print_section("Testing remove_duplicates")
    clean_df_duplicates, report_duplicates = remove_duplicates(df)
    print("Report for duplicate removal:")
    print_report(report_duplicates)
    compare_dataframes(df, clean_df_duplicates, "Duplicate Removal")

    print_section("Testing encode_categorical")

    clean_df_label, report_label = encode_categorical(
        df, columns=['status'], method='label')
    print("Report for label encoding:")
    print_report(report_label)
    compare_dataframes(df[['status']], clean_df_label[[
                       'status']], "Label Encoding")

    clean_df_onehot, report_onehot = encode_categorical(
        df, columns=['status'], method='onehot')
    print("Report for one-hot encoding:")
    print_report(report_onehot)

    onehot_cols = [
        col for col in clean_df_onehot.columns if col.startswith('status_')]
    print("\nOne-hot encoded columns:")
    print(clean_df_onehot[onehot_cols].head())

    print_section("Testing scale_numeric_features")

    clean_df_scale, report_scale = scale_numeric_features(
        df, columns=['income', 'score'], method='standard')
    print("Report for standard scaling:")
    print_report(report_scale)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    sns.histplot(df['income'].dropna(), kde=True)
    plt.title("Income Before Scaling")
    plt.subplot(1, 2, 2)
    sns.histplot(clean_df_scale['income'].dropna(), kde=True)
    plt.title("Income After Standard Scaling")
    plt.tight_layout()
    plt.show()

    print_section("Testing standardize_column_names")
    clean_df_names, report_names = standardize_column_names(df, case='snake')
    print("Report for column name standardization:")
    print_report(report_names)
    print("\nOriginal column names vs. standardized names:")
    for orig, new in report_names['column_mappings'].items():
        print(f"{orig:20} -> {new}")

    print_section("Testing correct_inconsistent_values")

    print("Category distribution before correction:")
    visualize_categorical_distribution(
        df, 'category', "Categories Before Correction")

    clean_df_inconsistent, report_inconsistent = correct_inconsistent_values(
        df, columns=['category'], method='cluster')
    print("Report for inconsistent value correction:")
    print_report(report_inconsistent)

    print("Category distribution after correction:")
    visualize_categorical_distribution(
        clean_df_inconsistent, 'category', "Categories After Correction")

    print_section("Testing comprehensive clean_dataframe")

    test_df = original_df.copy()

    final_df, final_report = clean_dataframe(
        test_df,
        null_method='mean',
        fix_numeric=True,
        fix_dates=True,
        handle_text=True,
        detect_outliers=True,
        outlier_action='remove',
        remove_dups=True,
        standardize_columns=True,
        encode_cats=True,
        scale_numeric=True,
        correct_inconsistent=True,
        column_case='snake'
    )

    print("Report for comprehensive cleaning:")
    print_report(final_report)

    print("\nFinal data shape:", final_df.shape)
    print("\nFinal data types:")
    print(final_df.dtypes)

    print("\nSample of final cleaned data:")
    print(final_df.head())

    print("\nMemory usage comparison:")
    print(f"Original: {final_report['memory_usage']['before']:.2f} MB")
    print(f"Cleaned: {final_report['memory_usage']['after']:.2f} MB")

    print("\nProcessing time for each operation:")
    for op, time_taken in final_report['processing_time']['operations'].items():
        print(f"{op}: {time_taken:.2f} seconds")

    print("\nTotal processing time:",
          final_report['processing_time']['total_seconds'], "seconds")

    print_section("Test Complete")
    print("All data_cleanup features have been tested successfully!")


if __name__ == "__main__":
    main()
