import pandas as pd
import numpy as np
import concurrent.futures
import time
from typing import Dict, List, Tuple, Union, Optional, Any


def handle_null_values(df: pd.DataFrame, method: str = 'nan', columns: Optional[List[str]] = None,
                       numeric_fill: Optional[Union[int, float]] = None,
                       categorical_fill: str = 'unknown',
                       date_fill: Optional[str] = None,
                       inplace: bool = False) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Check for null values and replace them with specified values based on data type.

    Parameters:
    -----------
    df : pandas.DataFrame
        The dataframe to check for null values
    method : str
        Method to handle null values: 'nan', 'zero', 'mean', 'median', 'mode', 'custom'
    columns : list, optional
        List of column names to check. If None, check all columns
    numeric_fill : int or float, optional
        Custom value to fill numeric nulls when method='custom'
    categorical_fill : str, optional
        Value to fill categorical nulls when method='custom'
    date_fill : str, optional
        Value to fill datetime nulls (ISO format) when method='custom'
    inplace : bool, default False
        Whether to perform operation in-place on input df

    Returns:
    --------
    pandas.DataFrame
        The dataframe with null values handled
    dict
        A report of null values before and after handling
    """
    if columns is None:
        columns = df.columns.tolist()
    else:
        columns = [col for col in columns if col in df.columns]

    df_result = df if inplace else df.copy()

    null_counts = df_result[columns].isna().sum()
    null_report = {
        'before': {col: int(null_counts[col]) for col in columns},
        'replaced_values': {},
        'method_used': method
    }

    numeric_cols = df_result[columns].select_dtypes(include=['number']).columns
    datetime_cols = df_result[columns].select_dtypes(
        include=['datetime']).columns
    categorical_cols = list(
        set(columns) - set(numeric_cols) - set(datetime_cols))

    if method == 'zero':
        if numeric_cols.any():
            df_result[numeric_cols] = df_result[numeric_cols].fillna(0)
            for col in numeric_cols:
                if null_counts[col] > 0:
                    null_report['replaced_values'][col] = {
                        'method': 'zero', 'count': int(null_counts[col])}

        if categorical_cols:
            df_result[categorical_cols] = df_result[categorical_cols].fillna(
                'unknown')
            for col in categorical_cols:
                if null_counts[col] > 0:
                    null_report['replaced_values'][col] = {
                        'method': 'unknown', 'count': int(null_counts[col])}

    elif method == 'mean' and numeric_cols.any():
        for col in numeric_cols:
            if null_counts[col] > 0:
                mean_val = df_result[col].mean()
                df_result[col] = df_result[col].fillna(mean_val)
                null_report['replaced_values'][col] = {
                    'method': 'mean', 'value': float(mean_val), 'count': int(null_counts[col])}

    elif method == 'median' and numeric_cols.any():
        for col in numeric_cols:
            if null_counts[col] > 0:
                median_val = df_result[col].median()
                df_result[col] = df_result[col].fillna(median_val)
                null_report['replaced_values'][col] = {
                    'method': 'median', 'value': float(median_val), 'count': int(null_counts[col])}

    elif method == 'mode':
        for col in columns:
            if null_counts[col] > 0:
                mode_val = df_result[col].mode()[0]
                df_result[col] = df_result[col].fillna(mode_val)
                mode_display = str(mode_val) if not pd.isna(
                    mode_val) else 'NaN'
                null_report['replaced_values'][col] = {
                    'method': 'mode', 'value': mode_display, 'count': int(null_counts[col])}

    elif method == 'custom':
        if numeric_cols.any() and numeric_fill is not None:
            df_result[numeric_cols] = df_result[numeric_cols].fillna(
                numeric_fill)
            for col in numeric_cols:
                if null_counts[col] > 0:
                    null_report['replaced_values'][col] = {
                        'method': 'custom', 'value': numeric_fill, 'count': int(null_counts[col])}

        if categorical_cols and categorical_fill is not None:
            df_result[categorical_cols] = df_result[categorical_cols].fillna(
                categorical_fill)
            for col in categorical_cols:
                if null_counts[col] > 0:
                    null_report['replaced_values'][col] = {
                        'method': 'custom', 'value': categorical_fill, 'count': int(null_counts[col])}

        if datetime_cols.any() and date_fill is not None:
            try:
                date_value = pd.to_datetime(date_fill)
                df_result[datetime_cols] = df_result[datetime_cols].fillna(
                    date_value)
                for col in datetime_cols:
                    if null_counts[col] > 0:
                        null_report['replaced_values'][col] = {
                            'method': 'custom', 'value': date_fill, 'count': int(null_counts[col])}
            except:
                pass
    else:
        for col in columns:
            if null_counts[col] > 0:
                null_report['replaced_values'][col] = {
                    'method': 'left as NaN', 'count': int(null_counts[col])}

    null_report['after'] = {
        col: int(df_result[col].isna().sum()) for col in columns}

    return df_result, null_report


def _process_numeric_chunk(chunk, number_word_map):
    """Helper function to process a chunk of data for numeric conversion."""
    if chunk.dtype == 'object':
        chunk = chunk.astype(str)
        for word, number in number_word_map.items():
            pattern = r'\b' + word + r'\b'
            chunk = chunk.str.replace(pattern, number, case=False, regex=True)
        chunk = chunk.str.replace(r'[$£€¥,]', '', regex=True)
        chunk = chunk.str.replace(r'(\d+)%', r'\1', regex=True)
        chunk = chunk.str.replace(r'(\d+)y\/o', r'\1', regex=True)
        chunk = chunk.str.extract(r'(-?\d+\.?\d*)')[0]
    return pd.to_numeric(chunk, errors='coerce')


def fix_numeric_datatypes(df: pd.DataFrame, columns: Optional[List[str]] = None,
                          threshold: float = 0.5, parallel: bool = True,
                          inplace: bool = False) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Convert string/object columns to numeric types when possible.

    Parameters:
    -----------
    df : pandas.DataFrame
        The dataframe to process
    columns : list, optional
        List of columns to convert. If None, auto-detect potential numeric columns
    threshold : float
        Minimum proportion of successful conversions to consider a column as numeric
    parallel : bool
        Whether to use parallel processing for large datasets
    inplace : bool
        Whether to perform operation in-place on input df

    Returns:
    --------
    pandas.DataFrame
        The dataframe with fixed numeric types
    dict
        A report about the conversions
    """
    df_result = df if inplace else df.copy()

    number_word_map = {
        'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4',
        'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9',
        'ten': '10', 'twenty': '20', 'thirty': '30', 'forty': '40', 'fifty': '50',
        'sixty': '60', 'seventy': '70', 'eighty': '80', 'ninety': '90',
        'hundred': '100', 'thousand': '1000', 'million': '1000000', 'billion': '1000000000'
    }

    if columns is None:
        columns = []
        for col in df_result.columns:
            if pd.api.types.is_numeric_dtype(df_result[col]):
                columns.append(col)
                continue
            if pd.api.types.is_object_dtype(df_result[col]):
                sample = df_result[col].dropna().head(100)
                if len(sample) > 0:
                    success_count = pd.to_numeric(
                        sample, errors='coerce').notna().sum()
                    if success_count / len(sample) >= threshold:
                        columns.append(col)

    conversion_report = {
        'successful_conversions': {},
        'failed_conversions': {},
        'original_dtypes': {col: str(df_result[col].dtype) for col in columns},
        'final_dtypes': {}
    }

    use_parallel = parallel and len(df_result) > 10000 and len(columns) > 5

    if use_parallel:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = {}
            for col in columns:
                if pd.api.types.is_numeric_dtype(df_result[col]):
                    continue

                original_data = df_result[col].copy()
                non_null_count = original_data.notna().sum()

                results[col] = executor.submit(
                    _process_numeric_chunk, original_data, number_word_map)

            for col, future in results.items():
                df_result[col] = future.result()
                successful = df_result[col].notna().sum()
                failed = non_null_count - successful

                conversion_report['successful_conversions'][col] = int(
                    successful)
                conversion_report['failed_conversions'][col] = int(failed)
                conversion_report['final_dtypes'][col] = str(
                    df_result[col].dtype)
    else:
        for col in columns:
            if pd.api.types.is_numeric_dtype(df_result[col]):
                continue

            original_data = df_result[col].copy()
            non_null_count = original_data.notna().sum()

            df_result[col] = _process_numeric_chunk(
                original_data, number_word_map)

            successful = df_result[col].notna().sum()
            failed = non_null_count - successful

            conversion_report['successful_conversions'][col] = int(successful)
            conversion_report['failed_conversions'][col] = int(failed)
            conversion_report['final_dtypes'][col] = str(df_result[col].dtype)

    return df_result, conversion_report


def detect_and_fix_date_columns(df: pd.DataFrame, columns: Optional[List[str]] = None,
                                threshold: float = 0.5, inplace: bool = False) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Detect and convert date/datetime columns to proper types.

    Parameters:
    -----------
    df : pandas.DataFrame
        The dataframe to process
    columns : list, optional
        List of column names to check. If None, check all object columns
    threshold : float
        Minimum proportion of successful conversions to consider a column as datetime
    inplace : bool
        Whether to perform operation in-place on input df

    Returns:
    --------
    pandas.DataFrame
        The dataframe with fixed date types
    dict
        A report about the conversions
    """
    df_result = df if inplace else df.copy()

    if columns is None:
        columns = df_result.select_dtypes(include=['object']).columns.tolist()

    date_report = {
        'conversions': {},
        'original_dtypes': {col: str(df_result[col].dtype) for col in columns},
        'final_dtypes': {}
    }

    date_formats = [
        '%Y-%m-%d', '%m/%d/%Y', '%d/%m/%Y', '%Y/%m/%d',
        '%m-%d-%Y', '%d-%m-%Y', '%b %d, %Y', '%d %b %Y',
        '%Y-%m-%d %H:%M:%S', '%m/%d/%Y %H:%M:%S', '%d/%m/%Y %H:%M:%S',
        '%Y%m%d', '%d-%b-%Y', '%d%b%Y', '%b-%d-%Y'
    ]

    for col in columns:
        if pd.api.types.is_datetime64_dtype(df_result[col]):
            date_report['conversions'][col] = {
                'status': 'already_datetime',
                'format': 'native'
            }
            continue

        if pd.api.types.is_numeric_dtype(df_result[col]):
            continue

        sample = df_result[col].dropna().head(100).astype(str)
        if len(sample) == 0:
            continue

        converted = pd.to_datetime(sample, errors='coerce')
        success_rate = converted.notna().mean()

        if success_rate >= threshold:
            try:
                df_result[col] = pd.to_datetime(
                    df_result[col], errors='coerce')
                date_report['conversions'][col] = {
                    'status': 'success',
                    'format': 'auto',
                    'success_rate': float(success_rate)
                }
            except:
                best_format = None
                best_success = 0

                for fmt in date_formats:
                    try:
                        converted = pd.to_datetime(
                            sample, format=fmt, errors='coerce')
                        curr_success = converted.notna().mean()
                        if curr_success > best_success:
                            best_success = curr_success
                            best_format = fmt
                    except:
                        continue

                if best_format and best_success >= threshold:
                    try:
                        df_result[col] = pd.to_datetime(
                            df_result[col], format=best_format, errors='coerce')
                        date_report['conversions'][col] = {
                            'status': 'success',
                            'format': best_format,
                            'success_rate': float(best_success)
                        }
                    except:
                        date_report['conversions'][col] = {
                            'status': 'failed',
                            'reason': 'conversion_error'
                        }
                else:
                    date_report['conversions'][col] = {
                        'status': 'failed',
                        'reason': 'no_suitable_format',
                        'best_guess': best_format,
                        'best_success': float(best_success)
                    }

    date_report['final_dtypes'] = {col: str(df_result[col].dtype)
                                   for col in date_report['conversions'].keys()}

    return df_result, date_report


def handle_outliers(df: pd.DataFrame, columns: Optional[List[str]] = None,
                    method: str = 'iqr', threshold: float = 1.5,
                    action: str = 'report', inplace: bool = False) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Detect and handle outliers in numeric columns.

    Parameters:
    -----------
    df : pandas.DataFrame
        The dataframe to process
    columns : list, optional
        List of numeric column names to check. If None, check all numeric columns
    method : str
        Method to detect outliers: 'iqr' (Interquartile Range), 'zscore', or 'percentile'
    threshold : float
        Threshold for outlier detection (1.5 for IQR, 3 for z-score)
    action : str
        How to handle outliers: 'report', 'remove', 'clip', or 'winsorize'
    inplace : bool
        Whether to perform operation in-place on input df

    Returns:
    --------
    pandas.DataFrame
        The dataframe with outliers handled
    dict
        A report about the outliers
    """
    df_result = df if inplace else df.copy()

    if columns is None:
        columns = df_result.select_dtypes(include=['number']).columns.tolist()
    else:
        columns = [
            col for col in columns if pd.api.types.is_numeric_dtype(df_result[col])]

    outlier_report = {
        'detection_method': method,
        'threshold': threshold,
        'action': action,
        'columns': {}
    }

    for col in columns:
        if df_result[col].isna().all():
            continue

        col_data = df_result[col].dropna()
        outlier_mask = pd.Series(False, index=df_result.index)

        if method == 'iqr':
            Q1 = col_data.quantile(0.25)
            Q3 = col_data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            outlier_mask = (df_result[col] < lower_bound) | (
                df_result[col] > upper_bound)
            bounds = {'lower': float(lower_bound), 'upper': float(upper_bound)}

        elif method == 'zscore':
            mean = col_data.mean()
            std = col_data.std()
            z_scores = (df_result[col] - mean) / std
            outlier_mask = z_scores.abs() > threshold
            bounds = {
                'lower': float(mean - threshold * std),
                'upper': float(mean + threshold * std)
            }

        elif method == 'percentile':
            lower_percentile = threshold
            upper_percentile = 100 - threshold
            lower_bound = col_data.quantile(lower_percentile / 100)
            upper_bound = col_data.quantile(upper_percentile / 100)
            outlier_mask = (df_result[col] < lower_bound) | (
                df_result[col] > upper_bound)
            bounds = {'lower': float(lower_bound), 'upper': float(upper_bound)}

        outlier_count = outlier_mask.sum()
        total_values = df_result[col].notna().sum()
        outlier_percentage = (outlier_count / total_values) * \
            100 if total_values > 0 else 0

        outlier_report['columns'][col] = {
            'outlier_count': int(outlier_count),
            'total_values': int(total_values),
            'outlier_percentage': round(outlier_percentage, 2),
            'bounds': bounds
        }

        if outlier_count > 0:
            if action == 'remove':
                df_result.loc[outlier_mask, col] = np.nan
                outlier_report['columns'][col]['action_taken'] = 'removed'

            elif action == 'clip':
                if method in ['iqr', 'zscore', 'percentile']:
                    df_result[col] = df_result[col].clip(
                        lower=bounds['lower'], upper=bounds['upper'])
                    outlier_report['columns'][col]['action_taken'] = 'clipped'

            elif action == 'winsorize':
                if outlier_mask.any():
                    if method in ['iqr', 'zscore', 'percentile']:
                        df_result.loc[(df_result[col] < bounds['lower']) &
                                      df_result[col].notna(), col] = bounds['lower']
                        df_result.loc[(df_result[col] > bounds['upper']) &
                                      df_result[col].notna(), col] = bounds['upper']
                        outlier_report['columns'][col]['action_taken'] = 'winsorized'
            else:
                outlier_report['columns'][col]['action_taken'] = 'reported'
                if outlier_count > 0:
                    sample_outliers = df_result.loc[outlier_mask, col].head(
                        5).tolist()
                    outlier_report['columns'][col]['example_outliers'] = sample_outliers

    return df_result, outlier_report


def clean_text_columns(df: pd.DataFrame, columns: Optional[List[str]] = None,
                       lowercase: bool = True, strip_spaces: bool = True,
                       remove_special_chars: bool = False,
                       special_char_pattern: str = r'[^\w\s]',
                       inplace: bool = False) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Clean and normalize text data in string columns.

    Parameters:
    -----------
    df : pandas.DataFrame
        The dataframe to process
    columns : list, optional
        List of column names to clean. If None, clean all object columns
    lowercase : bool
        Whether to convert text to lowercase
    strip_spaces : bool
        Whether to strip leading/trailing spaces and normalize internal spacing
    remove_special_chars : bool
        Whether to remove special characters
    special_char_pattern : str
        Regex pattern for special characters to remove
    inplace : bool
        Whether to perform operation in-place on input df

    Returns:
    --------
    pandas.DataFrame
        The dataframe with cleaned text
    dict
        A report about the cleaning operations
    """
    df_result = df if inplace else df.copy()

    if columns is None:
        columns = df_result.select_dtypes(include=['object']).columns.tolist()
    else:
        columns = [col for col in columns if pd.api.types.is_string_dtype(df_result[col])
                   or pd.api.types.is_object_dtype(df_result[col])]

    text_report = {
        'operations': {
            'lowercase': lowercase,
            'strip_spaces': strip_spaces,
            'remove_special_chars': remove_special_chars,
            'special_char_pattern': special_char_pattern if remove_special_chars else None
        },
        'columns': {}
    }

    for col in columns:
        if df_result[col].isna().all():
            continue

        df_result[col] = df_result[col].astype(str)
        original_values = df_result[col].copy()
        changes = 0

        if lowercase:
            df_result[col] = df_result[col].str.lower()

        if strip_spaces:
            df_result[col] = df_result[col].str.strip()
            df_result[col] = df_result[col].str.replace(
                r'\s+', ' ', regex=True)

        if remove_special_chars:
            df_result[col] = df_result[col].str.replace(
                special_char_pattern, '', regex=True)

        changes = (original_values != df_result[col]).sum()
        change_percentage = (changes / len(df_result)) * 100

        text_report['columns'][col] = {
            'rows_changed': int(changes),
            'change_percentage': round(change_percentage, 2),
            'example': df_result[col].head(3).tolist() if changes > 0 else []
        }

        mask = original_values.isna()
        if mask.any():
            df_result.loc[mask, col] = np.nan

    return df_result, text_report


def remove_duplicates(df: pd.DataFrame, subset: Optional[List[str]] = None,
                      keep: str = 'first', inplace: bool = False) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Remove duplicate rows from a DataFrame.

    Parameters:
    -----------
    df : pandas.DataFrame
        The dataframe to process
    subset : list, optional
        List of columns to consider for identifying duplicates. If None, use all columns
    keep : str, default 'first'
        'first' - Keep first occurrence of duplicates
        'last' - Keep last occurrence of duplicates
        False - Drop all duplicates
    inplace : bool, default False
        Whether to perform operation in-place on input df

    Returns:
    --------
    pandas.DataFrame
        The dataframe with duplicates removed
    dict
        A report about the duplicates removed
    """
    df_result = df if inplace else df.copy()

    if subset is None:
        duplicate_mask = df_result.duplicated(keep=False)
    else:

        valid_subset = [col for col in subset if col in df_result.columns]
        if not valid_subset:

            duplicate_mask = df_result.duplicated(keep=False)
        else:
            duplicate_mask = df_result.duplicated(
                subset=valid_subset, keep=False)

    duplicate_count = duplicate_mask.sum()

    duplicate_report = {
        'total_duplicates_found': int(duplicate_count),
        'percentage_duplicates': round((duplicate_count / len(df_result)) * 100, 2) if len(df_result) > 0 else 0,
        'action': f"kept_{keep}" if keep else "removed_all"
    }

    if duplicate_count > 0:

        sample_size = min(5, duplicate_count)
        if subset is None:
            duplicate_samples = df_result[duplicate_mask].head(sample_size)
        else:
            valid_subset = [col for col in subset if col in df_result.columns]
            if not valid_subset:
                duplicate_samples = df_result[duplicate_mask].head(sample_size)
            else:
                duplicate_samples = df_result[duplicate_mask].head(sample_size)

        duplicate_report['sample_duplicates'] = duplicate_samples.to_dict(
            'records')

        original_shape = df_result.shape
        if subset is None:
            df_result.drop_duplicates(keep=keep, inplace=True)
        else:
            valid_subset = [col for col in subset if col in df_result.columns]
            if not valid_subset:
                df_result.drop_duplicates(keep=keep, inplace=True)
            else:
                df_result.drop_duplicates(
                    subset=valid_subset, keep=keep, inplace=True)

        new_shape = df_result.shape
        rows_removed = original_shape[0] - new_shape[0]

        duplicate_report['rows_removed'] = int(rows_removed)

    return df_result, duplicate_report


def encode_categorical(df: pd.DataFrame, columns: Optional[List[str]] = None,
                       method: str = 'onehot', max_categories: int = 20,
                       inplace: bool = False) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Encode categorical variables using specified method.

    Parameters:
    -----------
    df : pandas.DataFrame
        The dataframe to process
    columns : list, optional
        List of columns to encode. If None, auto-detect categorical columns
    method : str
        Encoding method: 'onehot', 'label', or 'ordinal'
    max_categories : int
        Maximum number of unique categories for one-hot encoding
    inplace : bool
        Whether to perform operation in-place on input df

    Returns:
    --------
    pandas.DataFrame
        The dataframe with encoded categories
    dict
        A report about the encoding operations
    """
    df_result = df if inplace else df.copy()

    if columns is None:

        columns = []
        for col in df_result.select_dtypes(include=['object']).columns:
            n_unique = df_result[col].nunique()
            if n_unique < max_categories and n_unique > 1:
                columns.append(col)

    encoding_report = {
        'method': method,
        'encoded_columns': {}
    }

    if method == 'onehot':
        for col in columns:
            if df_result[col].nunique() > max_categories:
                encoding_report['encoded_columns'][col] = {
                    'status': 'skipped',
                    'reason': f'Too many categories ({df_result[col].nunique()} > {max_categories})'
                }
                continue

            dummies = pd.get_dummies(
                df_result[col], prefix=col, prefix_sep='_')

            df_result = pd.concat([df_result, dummies], axis=1)

            df_result.drop(col, axis=1, inplace=True)

            encoding_report['encoded_columns'][col] = {
                'status': 'success',
                'categories': dummies.columns.tolist(),
                'orig_nunique': int(dummies.shape[1])
            }

    elif method == 'label':
        from sklearn.preprocessing import LabelEncoder

        for col in columns:
            try:

                le = LabelEncoder()

                na_mask = df_result[col].isna()
                if na_mask.any():
                    df_result.loc[~na_mask, col] = le.fit_transform(
                        df_result.loc[~na_mask, col])
                else:
                    df_result[col] = le.fit_transform(df_result[col])

                encoding_report['encoded_columns'][col] = {
                    'status': 'success',
                    'mapping': {str(orig): int(encoded) for orig, encoded in
                                zip(le.classes_, range(len(le.classes_)))},
                    'orig_nunique': int(len(le.classes_))
                }
            except Exception as e:
                encoding_report['encoded_columns'][col] = {
                    'status': 'failed',
                    'reason': str(e)
                }

    elif method == 'ordinal':

        for col in columns:
            try:
                categories = df_result[col].dropna().unique()
                category_map = {cat: i for i, cat in enumerate(categories)}

                na_mask = df_result[col].isna()
                df_result.loc[~na_mask, col] = df_result.loc[~na_mask, col].map(
                    category_map)

                encoding_report['encoded_columns'][col] = {
                    'status': 'success',
                    'mapping': {str(orig): int(encoded) for orig, encoded in category_map.items()},
                    'orig_nunique': int(len(categories))
                }
            except Exception as e:
                encoding_report['encoded_columns'][col] = {
                    'status': 'failed',
                    'reason': str(e)
                }

    return df_result, encoding_report


def scale_numeric_features(df: pd.DataFrame, columns: Optional[List[str]] = None,
                           method: str = 'standard', feature_range: Tuple[float, float] = (0, 1),
                           inplace: bool = False) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Scale numeric features using specified method.

    Parameters:
    -----------
    df : pandas.DataFrame
        The dataframe to process
    columns : list, optional
        List of columns to scale. If None, scale all numeric columns
    method : str
        Scaling method: 'standard', 'minmax', 'robust', or 'log'
    feature_range : tuple
        Min/max values for MinMaxScaler
    inplace : bool
        Whether to perform operation in-place on input df

    Returns:
    --------
    pandas.DataFrame
        The dataframe with scaled features
    dict
        A report about the scaling operations
    """
    df_result = df if inplace else df.copy()

    if columns is None:
        columns = df_result.select_dtypes(include=['number']).columns.tolist()

    scaling_report = {
        'method': method,
        'scaled_columns': {}
    }

    if method == 'standard':
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()

        for col in columns:
            na_mask = df_result[col].isna()
            if na_mask.all():
                scaling_report['scaled_columns'][col] = {
                    'status': 'skipped',
                    'reason': 'All values are NaN'
                }
                continue

            data = df_result.loc[~na_mask, col].values.reshape(-1, 1)

            try:
                scaled_data = scaler.fit_transform(data)

                mean = float(scaler.mean_[0])
                var = float(scaler.var_[0])

                df_result.loc[~na_mask, col] = scaled_data.flatten()

                scaling_report['scaled_columns'][col] = {
                    'status': 'success',
                    'original_mean': mean,
                    'original_var': var,
                    'new_mean': float(df_result[col].mean()),
                    'new_var': float(df_result[col].var())
                }
            except Exception as e:
                scaling_report['scaled_columns'][col] = {
                    'status': 'failed',
                    'reason': str(e)
                }

    elif method == 'minmax':
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler(feature_range=feature_range)

        for col in columns:
            na_mask = df_result[col].isna()
            if na_mask.all():
                scaling_report['scaled_columns'][col] = {
                    'status': 'skipped',
                    'reason': 'All values are NaN'
                }
                continue

            data = df_result.loc[~na_mask, col].values.reshape(-1, 1)

            try:
                scaled_data = scaler.fit_transform(data)

                min_val = float(scaler.data_min_[0])
                max_val = float(scaler.data_max_[0])

                df_result.loc[~na_mask, col] = scaled_data.flatten()

                scaling_report['scaled_columns'][col] = {
                    'status': 'success',
                    'original_min': min_val,
                    'original_max': max_val,
                    'new_min': float(df_result[col].min()),
                    'new_max': float(df_result[col].max())
                }
            except Exception as e:
                scaling_report['scaled_columns'][col] = {
                    'status': 'failed',
                    'reason': str(e)
                }

    elif method == 'robust':
        from sklearn.preprocessing import RobustScaler
        scaler = RobustScaler()

        for col in columns:
            na_mask = df_result[col].isna()
            if na_mask.all():
                scaling_report['scaled_columns'][col] = {
                    'status': 'skipped',
                    'reason': 'All values are NaN'
                }
                continue

            data = df_result.loc[~na_mask, col].values.reshape(-1, 1)

            try:
                scaled_data = scaler.fit_transform(data)

                df_result.loc[~na_mask, col] = scaled_data.flatten()

                scaling_report['scaled_columns'][col] = {
                    'status': 'success',
                    'center': float(scaler.center_[0]),
                    'scale': float(scaler.scale_[0]),
                    'new_median': float(df_result[col].median())
                }
            except Exception as e:
                scaling_report['scaled_columns'][col] = {
                    'status': 'failed',
                    'reason': str(e)
                }

    elif method == 'log':
        for col in columns:
            na_mask = df_result[col].isna()
            if na_mask.all():
                scaling_report['scaled_columns'][col] = {
                    'status': 'skipped',
                    'reason': 'All values are NaN'
                }
                continue

            min_val = df_result[col].min()
            if min_val <= 0:
                offset = abs(min_val) + 1
                df_result[col] = df_result[col] + offset
                has_offset = True
            else:
                has_offset = False

            try:
                df_result[col] = np.log(df_result[col])

                scaling_report['scaled_columns'][col] = {
                    'status': 'success',
                    'transform': 'log',
                    'applied_offset': has_offset,
                    'offset_value': float(offset) if has_offset else 0
                }
            except Exception as e:
                scaling_report['scaled_columns'][col] = {
                    'status': 'failed',
                    'reason': str(e)
                }

    return df_result, scaling_report


def standardize_column_names(df: pd.DataFrame, case: str = 'snake',
                             prefix: str = '', suffix: str = '',
                             replace_spaces: bool = True,
                             inplace: bool = False) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Standardize column names using consistent naming conventions.

    Parameters:
    -----------
    df : pandas.DataFrame
        The dataframe to process
    case : str
        Naming convention: 'snake', 'camel', 'pascal', or 'kebab'
    prefix : str
        Prefix to add to column names
    suffix : str
        Suffix to add to column names
    replace_spaces : bool
        Whether to replace spaces with underscores
    inplace : bool
        Whether to perform operation in-place on input df

    Returns:
    --------
    pandas.DataFrame
        The dataframe with standardized column names
    dict
        A report about the renaming operations
    """
    import re

    df_result = df if inplace else df.copy()

    rename_report = {
        'column_mappings': {},
        'case': case,
        'prefix': prefix,
        'suffix': suffix
    }

    def convert_case(s, case_type):

        if replace_spaces:
            s = re.sub(r'[^\w\s]', '', s)
            s = re.sub(r'\s+', '_', s.strip())

        if case_type == 'snake':
            s = re.sub(r'([A-Z])', r'_\1', s).lower()
            s = re.sub(r'_+', '_', s)
            if s.startswith('_'):
                s = s[1:]
        elif case_type == 'camel':

            parts = re.split(r'[_\s]', s)
            s = parts[0].lower() + ''.join(p.capitalize() for p in parts[1:])
        elif case_type == 'pascal':

            s = ''.join(p.capitalize() for p in re.split(r'[_\s]', s))
        elif case_type == 'kebab':
            s = re.sub(r'([A-Z])', r'-\1', s).lower()
            s = re.sub(r'-+', '-', s)
            s = re.sub(r'[_\s]', '-', s)
            if s.startswith('-'):
                s = s[1:]

        return prefix + s + suffix

    new_names = {}
    for col in df_result.columns:
        new_col = convert_case(str(col), case)
        new_names[col] = new_col
        rename_report['column_mappings'][col] = new_col

    df_result.rename(columns=new_names, inplace=True)

    return df_result, rename_report


def correct_inconsistent_values(df: pd.DataFrame, columns: Optional[List[str]] = None,
                                method: str = 'cluster', similarity_threshold: float = 0.85,
                                mapping: Optional[Dict[str,
                                                       Dict[str, str]]] = None,
                                inplace: bool = False) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Correct inconsistent categorical values using string similarity or mapping.

    Parameters:
    -----------
    df : pandas.DataFrame
        The dataframe to process
    columns : list, optional
        List of columns to standardize. If None, check all object columns
    method : str
        Method for correction: 'cluster', 'mapping', or 'frequency'
    similarity_threshold : float
        Threshold for string similarity (for clustering method)
    mapping : dict, optional
        Dictionary mapping columns to value corrections ({col: {old_val: new_val}})
    inplace : bool
        Whether to perform operation in-place on input df

    Returns:
    --------
    pandas.DataFrame
        The dataframe with corrected values
    dict
        A report about the correction operations
    """
    from difflib import SequenceMatcher

    df_result = df if inplace else df.copy()

    if columns is None:
        columns = df_result.select_dtypes(include=['object']).columns.tolist()

    correction_report = {
        'method': method,
        'corrections': {}
    }

    def string_similarity(a, b):
        return SequenceMatcher(None, str(a).lower(), str(b).lower()).ratio()

    if method == 'cluster':
        for col in columns:

            if df_result[col].nunique() > 1000:
                correction_report['corrections'][col] = {
                    'status': 'skipped',
                    'reason': 'Too many unique values'
                }
                continue

            unique_vals = df_result[col].dropna().unique()
            if len(unique_vals) <= 1:
                continue

            clusters = {}
            unclustered = list(unique_vals)

            while unclustered:
                current = unclustered.pop(0)
                cluster = [current]

                i = 0
                while i < len(unclustered):
                    if string_similarity(current, unclustered[i]) >= similarity_threshold:
                        cluster.append(unclustered.pop(i))
                    else:
                        i += 1

                if len(cluster) > 1:

                    counts = df_result[df_result[col].isin(
                        cluster)][col].value_counts()
                    canonical = counts.index[0]
                    clusters[canonical] = cluster

            if clusters:
                replacements = {}
                for canonical, variants in clusters.items():
                    for variant in variants:
                        if variant != canonical:
                            replacements[variant] = canonical

                if replacements:
                    df_result[col] = df_result[col].replace(replacements)
                    correction_report['corrections'][col] = {
                        'status': 'success',
                        'replacements': replacements,
                        'unique_values': {
                            'before': int(len(unique_vals)),
                            'after': int(df_result[col].nunique())
                        }
                    }

    elif method == 'mapping':
        if mapping is None:
            correction_report['status'] = 'failed'
            correction_report['reason'] = 'No mapping dictionary provided'
            return df_result, correction_report

        for col in columns:
            if col not in mapping:
                continue

            col_mapping = mapping[col]
            df_result[col] = df_result[col].replace(col_mapping)

            correction_report['corrections'][col] = {
                'status': 'success',
                'replacements': col_mapping,
                'unique_values': {
                    'before': int(df[col].nunique()),
                    'after': int(df_result[col].nunique())
                }
            }

    elif method == 'frequency':
        for col in columns:

            if df_result[col].nunique() > 1000:
                correction_report['corrections'][col] = {
                    'status': 'skipped',
                    'reason': 'Too many unique values'
                }
                continue

            value_counts = df_result[col].value_counts()
            if len(value_counts) <= 1:
                continue

            replacements = {}
            examined = set()

            for val1 in value_counts.index:
                if val1 in examined:
                    continue

                examined.add(val1)
                current_cluster = {val1}

                for val2 in value_counts.index:
                    if val1 != val2 and val2 not in examined:
                        if string_similarity(val1, val2) >= similarity_threshold:
                            current_cluster.add(val2)
                            examined.add(val2)

                if len(current_cluster) > 1:
                    cluster_vals = list(current_cluster)
                    counts = df_result[df_result[col].isin(
                        cluster_vals)][col].value_counts()
                    most_frequent = counts.index[0]

                    for val in cluster_vals:
                        if val != most_frequent:
                            replacements[val] = most_frequent

            if replacements:
                df_result[col] = df_result[col].replace(replacements)
                correction_report['corrections'][col] = {
                    'status': 'success',
                    'replacements': replacements,
                    'unique_values': {
                        'before': int(value_counts.shape[0]),
                        'after': int(df_result[col].nunique())
                    }
                }

    return df_result, correction_report


def clean_dataframe(df: pd.DataFrame,
                    null_method: str = 'nan',
                    fix_numeric: bool = True,
                    fix_dates: bool = True,
                    handle_text: bool = True,
                    detect_outliers: bool = True,
                    remove_dups: bool = True,
                    standardize_columns: bool = False,
                    encode_cats: bool = False,
                    scale_numeric: bool = False,
                    correct_inconsistent: bool = False,
                    dup_subset: Optional[List[str]] = None,
                    outlier_method: str = 'iqr',
                    outlier_action: str = 'report',
                    encoding_method: str = 'onehot',
                    scaling_method: str = 'standard',
                    column_case: str = 'snake',
                    inconsistency_method: str = 'cluster',
                    parallel: bool = True,
                    inplace: bool = False) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Apply all cleaning functions in sequence with performance optimizations.

    Parameters:
    -----------
    df : pandas.DataFrame
        The dataframe to clean
    null_method : str
        Method to handle null values: 'nan', 'zero', 'mean', 'median', 'mode'
    fix_numeric : bool
        Whether to fix numeric data types
    fix_dates : bool
        Whether to detect and fix date columns
    handle_text : bool
        Whether to clean text columns
    detect_outliers : bool
        Whether to detect and report outliers
    remove_dups : bool
        Whether to remove duplicates
    standardize_columns : bool
        Whether to standardize column names
    encode_cats : bool
        Whether to encode categorical variables
    scale_numeric : bool
        Whether to scale numeric features
    correct_inconsistent : bool
        Whether to correct inconsistent values
    dup_subset : list, optional
        Columns to consider for duplicates
    outlier_method : str
        Method for outlier detection
    outlier_action : str
        Action to take for outliers
    encoding_method : str
        Method for encoding categorical variables
    scaling_method : str
        Method for scaling numeric features
    column_case : str
        Case style for column names
    inconsistency_method : str
        Method for correcting inconsistencies
    parallel : bool
        Whether to use parallel processing for large datasets
    inplace : bool
        Whether to perform operation in-place on input df

    Returns:
    --------
    pandas.DataFrame
        The cleaned dataframe
    dict
        A report of all cleaning operations
    """
    start_time = time.time()

    report = {
        'original_shape': df.shape,
        'operations_performed': [],
        'memory_usage': {
            'before': df.memory_usage(deep=True).sum() / (1024 * 1024),
        },
        'processing_time': {},
        'data_profile': {
            'numeric_columns': len(df.select_dtypes(include=['number']).columns),
            'text_columns': len(df.select_dtypes(include=['object']).columns),
            'datetime_columns': len(df.select_dtypes(include=['datetime']).columns),
            'bool_columns': len(df.select_dtypes(include=['bool']).columns),
        }
    }

    df_result = df if inplace else df.copy()
    operation_times = {}

    if standardize_columns:
        op_start = time.time()
        df_result, col_report = standardize_column_names(
            df_result, case=column_case, inplace=True)
        operation_times['column_standardization'] = time.time() - op_start
        report['column_standardization'] = col_report
        report['operations_performed'].append('column_standardization')

    if null_method != 'skip':
        op_start = time.time()
        df_result, null_report = handle_null_values(
            df_result, method=null_method, inplace=True)
        operation_times['null_handling'] = time.time() - op_start
        report['null_handling'] = null_report
        report['operations_performed'].append('null_handling')

    if fix_numeric:
        op_start = time.time()
        df_result, dtype_report = fix_numeric_datatypes(
            df_result, parallel=parallel, inplace=True)
        operation_times['datatype_handling'] = time.time() - op_start
        report['datatype_handling'] = dtype_report
        report['operations_performed'].append('datatype_handling')

    if fix_dates:
        op_start = time.time()
        df_result, date_report = detect_and_fix_date_columns(
            df_result, inplace=True)
        operation_times['date_handling'] = time.time() - op_start
        report['date_handling'] = date_report
        report['operations_performed'].append('date_handling')

    if handle_text:
        op_start = time.time()
        df_result, text_report = clean_text_columns(df_result, inplace=True)
        operation_times['text_cleaning'] = time.time() - op_start
        report['text_cleaning'] = text_report
        report['operations_performed'].append('text_cleaning')

    if correct_inconsistent:
        op_start = time.time()
        df_result, inconsistent_report = correct_inconsistent_values(
            df_result, method=inconsistency_method, inplace=True
        )
        operation_times['inconsistency_correction'] = time.time() - op_start
        report['inconsistency_correction'] = inconsistent_report
        report['operations_performed'].append('inconsistency_correction')

    if detect_outliers:
        op_start = time.time()
        df_result, outlier_report = handle_outliers(
            df_result, method=outlier_method, action=outlier_action, inplace=True
        )
        operation_times['outlier_handling'] = time.time() - op_start
        report['outlier_handling'] = outlier_report
        report['operations_performed'].append('outlier_handling')

    if remove_dups:
        op_start = time.time()
        df_result, dup_report = remove_duplicates(
            df_result, subset=dup_subset, inplace=True)
        operation_times['duplicate_handling'] = time.time() - op_start
        report['duplicate_handling'] = dup_report
        report['operations_performed'].append('duplicate_handling')

    if encode_cats:
        op_start = time.time()
        df_result, encoding_report = encode_categorical(
            df_result, method=encoding_method, inplace=True)
        operation_times['categorical_encoding'] = time.time() - op_start
        report['categorical_encoding'] = encoding_report
        report['operations_performed'].append('categorical_encoding')

    if scale_numeric:
        op_start = time.time()
        df_result, scaling_report = scale_numeric_features(
            df_result, method=scaling_method, inplace=True)
        operation_times['numeric_scaling'] = time.time() - op_start
        report['numeric_scaling'] = scaling_report
        report['operations_performed'].append('numeric_scaling')

    report['final_shape'] = df_result.shape
    report['memory_usage']['after'] = df_result.memory_usage(
        deep=True).sum() / (1024 * 1024)
    report['shape_change'] = {
        'rows': report['final_shape'][0] - report['original_shape'][0],
        'columns': report['final_shape'][1] - report['original_shape'][1]
    }
    report['data_profile_after'] = {
        'numeric_columns': len(df_result.select_dtypes(include=['number']).columns),
        'text_columns': len(df_result.select_dtypes(include=['object']).columns),
        'datetime_columns': len(df_result.select_dtypes(include=['datetime']).columns),
        'bool_columns': len(df_result.select_dtypes(include=['bool']).columns),
    }
    report['processing_time'] = {
        'total_seconds': round(time.time() - start_time, 2),
        'operations': {op: round(time_taken, 2) for op, time_taken in operation_times.items()}
    }

    return df_result, report
