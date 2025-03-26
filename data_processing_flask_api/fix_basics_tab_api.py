from data_processing.data_cleanup import (
    handle_null_values, remove_duplicates, correct_inconsistent_values,
    clean_dataframe, fix_numeric_datatypes, detect_and_fix_date_columns,
    clean_text_columns, handle_outliers, standardize_column_names
)
import os
import json
import pandas as pd
from flask import Blueprint, request, jsonify
import sys
import time
import pathlib
import numpy as np

data_cleaning_api = Blueprint('data_cleaning_api', __name__)

def get_cleaned_path(dataset_path):
    """Generate path for cleaned dataset by adding '_cleaned' before extension"""
    path_obj = pathlib.Path(dataset_path)
    stem = path_obj.stem
    suffix = path_obj.suffix
    parent = path_obj.parent
    cleaned_path = parent / f"{stem}_cleaned{suffix}"
    return str(cleaned_path)


def read_dataset(dataset_path):
    """Read dataset from path based on file extension"""
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at path: {dataset_path}")

    file_extension = os.path.splitext(dataset_path)[1].lower()

    if file_extension == '.csv':
        return pd.read_csv(dataset_path)
    elif file_extension in ['.xls', '.xlsx']:
        return pd.read_excel(dataset_path)
    elif file_extension == '.json':
        return pd.read_json(dataset_path)
    else:
        raise ValueError(f"Unsupported file format: {file_extension}")


def save_dataset(df, dataset_path):
    """Save dataset to path based on file extension"""
    file_extension = os.path.splitext(dataset_path)[1].lower()

    os.makedirs(os.path.dirname(os.path.abspath(dataset_path)), exist_ok=True)

    if file_extension == '.csv':
        df.to_csv(dataset_path, index=False)
    elif file_extension in ['.xls', '.xlsx']:
        df.to_excel(dataset_path, index=False)
    elif file_extension == '.json':
        df.to_json(dataset_path, orient='records')
    else:
        raise ValueError(f"Unsupported file format: {file_extension}")

    return dataset_path


@data_cleaning_api.route('/api/data-cleaning/validate-path', methods=['POST'])
def validate_path():
    """
    Validate if a dataset path exists and can be read
    """
    data = request.json
    dataset_path = data.get('dataset_path')

    if not dataset_path:
        return jsonify({'error': 'No dataset path provided'}), 400

    try:
        df = read_dataset(dataset_path)

        return jsonify({
            'status': 'success',
            'message': 'Dataset found and readable',
            'rows': df.shape[0],
            'columns': df.shape[1],
            'column_names': df.columns.tolist()
        }), 200

    except FileNotFoundError:
        return jsonify({'error': f'Dataset not found at path: {dataset_path}'}), 404
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        return jsonify({'error': f'Error reading dataset: {str(e)}'}), 500


@data_cleaning_api.route('/api/data-cleaning/missing-values', methods=['POST'])
def handle_missing_values():
    """
    Handle missing values in a dataset
    """
    data = request.json
    dataset_path = data.get('dataset_path')

    if not dataset_path:
        return jsonify({'error': 'No dataset path provided'}), 400

    fill_method = data.get('fill_method', 'nan')
    columns = data.get('columns')
    numeric_fill = data.get('numeric_fill_value')
    categorical_fill = data.get('categorical_fill_value', 'unknown')
    date_fill = data.get('date_fill_value')

    try:
        df = read_dataset(dataset_path)
        
        
        if columns:
            for col in columns:
                if col in df.columns:
                    print(f"Column {col} unique values before: {df[col].unique()}")
                    print(f"Column {col} null count before: {df[col].isna().sum()}")
        
        
        if columns and 'continent' in columns:
            
            df['continent'] = df['continent'].replace(['', 'null', 'NULL', 'None', 'NA', None], pd.NA)
        
        cleaned_df, report = handle_null_values(
            df=df,
            method=fill_method,
            columns=columns,
            numeric_fill=numeric_fill,
            categorical_fill=categorical_fill,
            date_fill=date_fill
        )
        
        cleaned_path = get_cleaned_path(dataset_path)
        save_dataset(cleaned_df, cleaned_path)
        
        
        rows_affected = 0
        columns_affected = 0
        for col in report['before']:
            if report['before'][col] > report['after'][col]:
                rows_affected += (report['before'][col] - report['after'][col])
                columns_affected += 1
        
        return jsonify({
            'status': 'success',
            'cleaned_dataset_path': cleaned_path,
            'report': report,
            'rows_affected': rows_affected,
            'columns_affected': columns_affected
        }), 200

    except FileNotFoundError:
        return jsonify({'error': f'Dataset not found at path: {dataset_path}'}), 404
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        return jsonify({'error': f'Error handling missing values: {str(e)}'}), 500


@data_cleaning_api.route('/api/data-cleaning/remove-duplicates', methods=['POST'])
def handle_duplicates():
    """
    Remove duplicate rows from a dataset
    """
    data = request.json
    dataset_path = data.get('dataset_path')

    if not dataset_path:
        return jsonify({'error': 'No dataset path provided'}), 400

    subset_columns = data.get('subset_columns')
    keep_strategy = data.get('keep_strategy', 'first')

    keep_map = {
        'first': 'first',
        'last': 'last',
        'none': False
    }
    keep = keep_map.get(keep_strategy, 'first')

    try:
        df = read_dataset(dataset_path)

        cleaned_df, report = remove_duplicates(
            df=df,
            subset=subset_columns,
            keep=keep
        )

        cleaned_path = get_cleaned_path(dataset_path)

        save_dataset(cleaned_df, cleaned_path)

        return jsonify({
            'status': 'success',
            'cleaned_dataset_path': cleaned_path,
            'report': report,
            'duplicates_found': report.get('total_duplicates_found', 0),
            'original_row_count': len(df),
            'new_row_count': len(cleaned_df)
        }), 200

    except FileNotFoundError:
        return jsonify({'error': f'Dataset not found at path: {dataset_path}'}), 404
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        return jsonify({'error': f'Error removing duplicates: {str(e)}'}), 500


@data_cleaning_api.route('/api/data-cleaning/fix-inconsistent-values', methods=['POST'])
def fix_inconsistencies():
    """
    Fix inconsistent values in string/categorical columns
    """
    data = request.json
    dataset_path = data.get('dataset_path')
    
    if not dataset_path:
        return jsonify({'error': 'No dataset path provided'}), 400
    
    columns = data.get('columns')
    correction_method = data.get('correction_method', 'automatic')
    similarity_threshold = data.get('similarity_threshold', 0.85)
    custom_mapping = data.get('custom_mapping')
    
    method_map = {
        'automatic': 'cluster',
        'frequency': 'frequency',
        'custom': 'mapping'
    }
    method = method_map.get(correction_method, 'cluster')
    
    try:
        df = read_dataset(dataset_path)
        
        
        if method == 'mapping' and custom_mapping:
            for col_name, mappings in custom_mapping.items():
                if col_name in df.columns:
                    
                    if 'null' in mappings or '' in mappings:
                        
                        null_value = mappings.get('null', mappings.get(''))
                        
                        df[col_name] = df[col_name].replace([None, pd.NA, np.nan, '', 'null', 'NULL', 'None', 'NA'], null_value)
        
        mapping_dict = None
        if method == 'mapping' and custom_mapping:
            mapping_dict = {col: {k: v for k, v in mappings.items()} 
                          for col, mappings in custom_mapping.items()}
        
        cleaned_df, report = correct_inconsistent_values(
            df=df, 
            columns=columns, 
            method=method,
            similarity_threshold=similarity_threshold,
            mapping=mapping_dict
        )
        
        cleaned_path = get_cleaned_path(dataset_path)
        save_dataset(cleaned_df, cleaned_path)
        
        return jsonify({
            'status': 'success',
            'cleaned_dataset_path': cleaned_path,
            'report': report,
            'columns_affected': len(report.get('corrections', {}))
        }), 200
        
    except FileNotFoundError:
        return jsonify({'error': f'Dataset not found at path: {dataset_path}'}), 404
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        return jsonify({'error': f'Error fixing inconsistent values: {str(e)}'}), 500


@data_cleaning_api.route('/api/data-cleaning/preview', methods=['POST'])
def preview_operations():
    """
    Preview the effects of data cleaning operations without applying them
    """
    data = request.json
    dataset_path = data.get('dataset_path')

    if not dataset_path:
        return jsonify({'error': 'No dataset path provided'}), 400

    operations = data.get('operations', [])

    if not operations:
        return jsonify({'error': 'No operations specified for preview'}), 400

    try:
        df = read_dataset(dataset_path)
        original_sample = df.head(5).to_dict('records')
        preview_df = df.copy()

        operation_reports = []

        for operation in operations:
            op_type = operation.get('type')
            op_params = operation.get('parameters', {})

            if op_type == 'missing_values':
                preview_df, report = handle_null_values(
                    df=preview_df,
                    method=op_params.get('fill_method', 'nan'),
                    columns=op_params.get('columns'),
                    numeric_fill=op_params.get('numeric_fill_value'),
                    categorical_fill=op_params.get(
                        'categorical_fill_value', 'unknown'),
                    date_fill=op_params.get('date_fill_value')
                )
                operation_reports.append({
                    'operation': 'missing_values',
                    'report': report
                })

            elif op_type == 'remove_duplicates':
                preview_df, report = remove_duplicates(
                    df=preview_df,
                    subset=op_params.get('subset_columns'),
                    keep=op_params.get('keep_strategy', 'first')
                )
                operation_reports.append({
                    'operation': 'remove_duplicates',
                    'report': report
                })

            elif op_type == 'fix_inconsistent_values':
                method_map = {
                    'automatic': 'cluster',
                    'frequency': 'frequency',
                    'custom': 'mapping'
                }
                method = method_map.get(op_params.get(
                    'correction_method', 'automatic'), 'cluster')

                mapping_dict = None
                if method == 'mapping' and op_params.get('custom_mapping'):
                    mapping_dict = {col: {k: v for k, v in mappings.items()}
                                    for col, mappings in op_params.get('custom_mapping').items()}

                preview_df, report = correct_inconsistent_values(
                    df=preview_df,
                    columns=op_params.get('columns'),
                    method=method,
                    similarity_threshold=op_params.get(
                        'similarity_threshold', 0.85),
                    mapping=mapping_dict
                )
                operation_reports.append({
                    'operation': 'fix_inconsistent_values',
                    'report': report
                })

        modified_sample = preview_df.head(5).to_dict('records')

        return jsonify({
            'status': 'success',
            'before_sample': original_sample,
            'after_sample': modified_sample,
            'operation_reports': operation_reports,
            'rows_before': df.shape[0],
            'rows_after': preview_df.shape[0],
            'columns_before': df.shape[1],
            'columns_after': preview_df.shape[1],
        }), 200

    except FileNotFoundError:
        return jsonify({'error': f'Dataset not found at path: {dataset_path}'}), 404
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        return jsonify({'error': f'Error generating preview: {str(e)}'}), 500

@data_cleaning_api.route('/api/data-cleaning/clean-all', methods=['POST'])
def clean_all():
    """
    Apply multiple cleaning operations in one API call using the clean_dataframe function
    """
    data = request.json
    dataset_path = data.get('dataset_path')

    if not dataset_path:
        return jsonify({'error': 'No dataset path provided'}), 400

    params = {
        'null_method': data.get('null_method', 'nan'),
        'fix_numeric': data.get('fix_numeric', True),
        'fix_dates': data.get('fix_dates', True),
        'handle_text': data.get('handle_text', True),
        'detect_outliers': data.get('detect_outliers', True),
        'remove_dups': data.get('remove_dups', True),
        'standardize_columns': data.get('standardize_columns', False),
        'encode_cats': data.get('encode_cats', False),
        'scale_numeric': data.get('scale_numeric', False),
        'correct_inconsistent': data.get('correct_inconsistent', False),
        'dup_subset': data.get('dup_subset'),
        'outlier_method': data.get('outlier_method', 'iqr'),
        'outlier_action': data.get('outlier_action', 'report'),
        'encoding_method': data.get('encoding_method', 'onehot'),
        'scaling_method': data.get('scaling_method', 'standard'),
        'column_case': data.get('column_case', 'snake'),
        'inconsistency_method': data.get('inconsistency_method', 'cluster'),
        'parallel': data.get('parallel', True)
    }

    try:
        df = read_dataset(dataset_path)
        start_time = time.time()

        cleaned_df, report = clean_dataframe(df=df, **params)

        cleaned_path = get_cleaned_path(dataset_path)

        save_dataset(cleaned_df, cleaned_path)

        processing_time = time.time() - start_time

        return jsonify({
            'status': 'success',
            'cleaned_dataset_path': cleaned_path,
            'report': report,
            'processing_time_seconds': processing_time,
            'original_shape': [df.shape[0], df.shape[1]],
            'final_shape': [cleaned_df.shape[0], cleaned_df.shape[1]]
        }), 200

    except FileNotFoundError:
        return jsonify({'error': f'Dataset not found at path: {dataset_path}'}), 404
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        return jsonify({'error': f'Error cleaning dataset: {str(e)}'}), 500


@data_cleaning_api.route('/api/data-cleaning/dataset-info', methods=['POST'])
def get_dataset_info():
    """Get information about a dataset at the specified path"""
    data = request.json
    dataset_path = data.get('dataset_path')

    if not dataset_path:
        return jsonify({'error': 'No dataset path provided'}), 400

    try:
        df = read_dataset(dataset_path)

        stats = {
            'rows': df.shape[0],
            'columns': df.shape[1],
            'column_names': df.columns.tolist(),
            'missing_values': df.isna().sum().to_dict(),
            'data_types': {col: str(dtype) for col, dtype in df.dtypes.items()},
            'sample_data': json.loads(df.head(5).to_json(orient='records'))
        }

        return jsonify(stats), 200

    except FileNotFoundError:
        return jsonify({'error': f'Dataset not found at path: {dataset_path}'}), 404
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        return jsonify({'error': f'Error getting dataset info: {str(e)}'}), 500