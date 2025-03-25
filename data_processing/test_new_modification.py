import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import time
import argparse
from pathlib import Path

# Add the parent directory to the path to import data_cleanup
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_processing.data_cleanup import *

def print_section(title):
    """Print a section header for better readability."""
    print("\n" + "="*80)
    print(f" {title} ".center(80, "="))
    print("="*80 + "\n")

def analyze_dataset(df, title="Dataset Analysis"):
    """Analyze and display key information about a dataset."""
    print_section(title)
    print(f"Shape: {df.shape} (rows, columns)")
    
    print("\nData Types:")
    for dtype, count in df.dtypes.value_counts().items():
        print(f"  {dtype}: {count} columns")
    
    print("\nMissing Values:")
    missing_cols = df.columns[df.isna().any()].tolist()
    if missing_cols:
        for col in missing_cols:
            missing_count = df[col].isna().sum()
            missing_pct = (missing_count / len(df)) * 100
            print(f"  {col}: {missing_count} values ({missing_pct:.2f}%)")
    else:
        print("  No missing values found")
    
    print("\nSample Data:")
    print(df.head(3))

def plot_data_profile(df, title="Data Profile"):
    """Create a basic data profile visualization."""
    # Setup plotting
    plt.style.use('ggplot')
    sns.set_theme(style="whitegrid")
    
    # Determine what to visualize based on dataframe content
    numeric_cols = df.select_dtypes(include=['number']).columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    date_cols = df.select_dtypes(include=['datetime']).columns
    
    # Plot profile of numeric columns (up to 4)
    if len(numeric_cols) > 0:
        cols_to_plot = numeric_cols[:min(4, len(numeric_cols))]
        fig, axes = plt.subplots(len(cols_to_plot), 2, figsize=(14, 4 * len(cols_to_plot)))
        if len(cols_to_plot) == 1:
            axes = np.array([axes])  # Ensure axes is 2D even with one row
        
        for i, col in enumerate(cols_to_plot):
            # Histogram
            sns.histplot(df[col].dropna(), kde=True, ax=axes[i, 0])
            axes[i, 0].set_title(f"Distribution of {col}")
            
            # Box plot
            sns.boxplot(x=df[col].dropna(), ax=axes[i, 1])
            axes[i, 1].set_title(f"Boxplot of {col}")
        
        plt.tight_layout()
        plt.savefig('numeric_profile.png')
        print(f"Saved numeric profile visualization to numeric_profile.png")
    
    # Plot profile of categorical columns (up to 4)
    if len(categorical_cols) > 0:
        cols_to_plot = categorical_cols[:min(4, len(categorical_cols))]
        plt.figure(figsize=(14, 4 * len(cols_to_plot)))
        
        for i, col in enumerate(cols_to_plot):
            plt.subplot(len(cols_to_plot), 1, i+1)
            value_counts = df[col].value_counts().head(10)  # Top 10 values
            sns.barplot(x=value_counts.index, y=value_counts.values)
            plt.title(f"Top values in {col}")
            plt.xticks(rotation=45)
            plt.tight_layout()
        
        plt.savefig('categorical_profile.png')
        print(f"Saved categorical profile visualization to categorical_profile.png")

def clean_csv_file(csv_path, output_path=None, inplace=False, config=None):
    """
    Clean a CSV file using data_cleanup functions.
    
    Parameters:
    -----------
    csv_path : str
        Path to the input CSV file
    output_path : str, optional
        Path to save the cleaned CSV (if None and not inplace, will use {original}_cleaned.csv)
    inplace : bool
        Whether to modify the original file or create a new one
    config : dict, optional
        Configuration settings for the cleaning process
        
    Returns:
    --------
    pandas.DataFrame
        The cleaned dataframe
    dict
        Report about the cleaning operations
    """
    # Set default configuration if none provided
    if config is None:
        config = {
            'null_method': 'mean',
            'fix_numeric': True,
            'fix_dates': True,
            'handle_text': True,
            'detect_outliers': True,
            'outlier_action': 'report',  # Just report outliers, don't remove them
            'remove_dups': True,
            'standardize_columns': False,  # Don't rename columns by default
            'encode_cats': False,  # Don't encode categories by default
            'scale_numeric': False,  # Don't scale by default
            'correct_inconsistent': True,
            'column_case': 'snake'
        }
    
    # Load the CSV file
    print(f"Loading CSV file: {csv_path}")
    start_time = time.time()
    try:
        df = pd.read_csv(csv_path, low_memory=False)
        load_time = time.time() - start_time
        print(f"Loaded {len(df)} rows and {len(df.columns)} columns in {load_time:.2f} seconds")
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        return None, None
    
    # Analyze the dataset before cleaning
    analyze_dataset(df, "Original Dataset")
    
    # Create visualization of original data
    try:
        plot_data_profile(df, "Original Data Profile")
    except Exception as e:
        print(f"Warning: Could not generate data profile visualizations: {e}")
    
    # Clean the dataframe
    print_section("Cleaning Data")
    print("Applying cleaning operations with the following configuration:")
    for k, v in config.items():
        print(f"  {k}: {v}")
    
    if inplace:
        final_df, report = clean_dataframe(df, inplace=True, **config)
    else:
        final_df, report = clean_dataframe(df, **config)
    
    # Analyze the cleaned dataset
    analyze_dataset(final_df, "Cleaned Dataset")
    
    # Create visualization of cleaned data
    try:
        plot_data_profile(final_df, "Cleaned Data Profile")
    except Exception as e:
        print(f"Warning: Could not generate data profile visualizations: {e}")
    
    # Save the cleaned dataframe
    if inplace:
        print(f"Saving cleaned data back to original file: {csv_path}")
        final_df.to_csv(csv_path, index=False)
    else:
        if output_path is None:
            # Create a default output path
            file_base = os.path.splitext(csv_path)[0]
            output_path = f"{file_base}_cleaned.csv"
        
        print(f"Saving cleaned data to new file: {output_path}")
        final_df.to_csv(output_path, index=False)
    
    # Save the cleaning report
    report_path = f"{os.path.splitext(output_path or csv_path)[0]}_report.json"
    import json
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    print(f"Saved cleaning report to: {report_path}")
    
    # Summary of changes
    print_section("Cleaning Summary")
    print(f"Rows: {df.shape[0]} → {final_df.shape[0]} ({final_df.shape[0] - df.shape[0]:+d})")
    print(f"Columns: {df.shape[1]} → {final_df.shape[1]} ({final_df.shape[1] - df.shape[1]:+d})")
    
    if 'null_handling' in report:
        total_nulls_before = sum(report['null_handling']['before'].values())
        total_nulls_after = sum(report['null_handling']['after'].values())
        print(f"Missing values: {total_nulls_before} → {total_nulls_after} ({total_nulls_after - total_nulls_before:+d})")
    
    if 'duplicate_handling' in report:
        print(f"Duplicates removed: {report['duplicate_handling'].get('removed_count', 0)}")
    
    if 'memory_usage' in report:
        before_mb = report['memory_usage']['before']
        after_mb = report['memory_usage']['after']
        print(f"Memory usage: {before_mb:.2f} MB → {after_mb:.2f} MB ({(after_mb - before_mb):+.2f} MB)")
    
    print(f"Total processing time: {report['processing_time']['total_seconds']:.2f} seconds")
    
    return final_df, report

def main():
    parser = argparse.ArgumentParser(description='Clean a CSV file using data_cleanup functions')
    parser.add_argument('file', help='Path to the CSV file to clean')
    parser.add_argument('--output', '-o', help='Path to save the cleaned CSV')
    parser.add_argument('--inplace', '-i', action='store_true', help='Modify the original file instead of creating a new one')
    parser.add_argument('--null-method', choices=['nan', 'zero', 'mean', 'median', 'mode', 'custom'], default='mean', 
                        help='Method to handle null values')
    parser.add_argument('--no-fix-numeric', action='store_true', help='Skip fixing numeric datatypes')
    parser.add_argument('--no-fix-dates', action='store_true', help='Skip detecting and fixing date columns')
    parser.add_argument('--no-handle-text', action='store_true', help='Skip cleaning text columns')
    parser.add_argument('--no-detect-outliers', action='store_true', help='Skip outlier detection')
    parser.add_argument('--outlier-action', choices=['report', 'remove', 'clip'], default='report',
                        help='Action to take for outliers')
    parser.add_argument('--no-remove-dups', action='store_true', help='Skip removing duplicates')
    parser.add_argument('--standardize-columns', action='store_true', help='Standardize column names')
    parser.add_argument('--encode-cats', action='store_true', help='Encode categorical columns')
    parser.add_argument('--scale-numeric', action='store_true', help='Scale numeric columns')
    parser.add_argument('--correct-inconsistent', action='store_true', help='Correct inconsistent values')
    parser.add_argument('--column-case', choices=['snake', 'camel', 'pascal', 'kebab'], default='snake',
                        help='Case style for standardized column names')
    
    args = parser.parse_args()
    
    # Check if file exists
    if not os.path.isfile(args.file):
        print(f"Error: File '{args.file}' does not exist")
        return
    
    # Build config from arguments
    config = {
        'null_method': args.null_method,
        'fix_numeric': not args.no_fix_numeric,
        'fix_dates': not args.no_fix_dates,
        'handle_text': not args.no_handle_text,
        'detect_outliers': not args.no_detect_outliers,
        'outlier_action': args.outlier_action,
        'remove_dups': not args.no_remove_dups,
        'standardize_columns': args.standardize_columns,
        'encode_cats': args.encode_cats,
        'scale_numeric': args.scale_numeric,
        'correct_inconsistent': args.correct_inconsistent,
        'column_case': args.column_case
    }
    
    # Clean the CSV file
    clean_csv_file(args.file, args.output, args.inplace, config)

# Example usage for simplicity (no need to use argparse if calling directly)
def simple_test():
    """Simple test function to clean a CSV file in the current directory."""
    # Look for any CSV file in the current directory
    csv_files = [f for f in os.listdir('.') if f.endswith('.csv')]
    
    if not csv_files:
        print("No CSV files found in the current directory.")
        print("Please place a CSV file in the same directory as this script.")
        return
    
    # Use the first CSV file found
    csv_path = csv_files[0]
    print(f"Found CSV file: {csv_path}")
    
    # Test in-place cleaning
    print_section("Testing In-Place Cleaning")
    # Make a backup of the original file first
    backup_path = f"{os.path.splitext(csv_path)[0]}_backup.csv"
    import shutil
    shutil.copy2(csv_path, backup_path)
    print(f"Created backup of original file: {backup_path}")
    
    # Clean in-place
    df_inplace, report_inplace = clean_csv_file(csv_path, inplace=True)
    
    # Test clean and save to new file
    print_section("Testing Clean and Save to New File")
    # Restore the original from backup
    shutil.copy2(backup_path, csv_path)
    print(f"Restored original file from backup")
    
    # Clean and save to new file
    output_path = f"{os.path.splitext(csv_path)[0]}_cleaned_new.csv"
    df_new, report_new = clean_csv_file(csv_path, output_path=output_path, inplace=False)
    
    print_section("Test Complete")
    print(f"Original file: {csv_path}")
    print(f"Backup file: {backup_path}")
    print(f"New cleaned file: {output_path}")
    print("\nBoth in-place cleaning and creating a new file have been tested.")
    

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # If arguments provided, run the command-line interface
        main()
    else:
        # Otherwise run the simple test on a CSV in current directory
        simple_test()