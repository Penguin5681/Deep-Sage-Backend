import pandas as pd
import os
from flask import Blueprint, jsonify, request
from flask_caching import Cache

data_api = Blueprint('data_api', __name__)

@data_api.route('/api/data/preview', methods=['POST'])
def preview_data():
    """
    Load the first N rows of CSV or Excel data.

    Request body:
    {
        "file_path": "/path/to/file.csv",  
        "n_rows": 10                      
    }

    Returns:
        JSON with data preview, column names, and metadata
    """
    data = request.json

    if not data or 'file_path' not in data:
        return jsonify({'error': 'file_path is required'}), 400

    file_path = data['file_path']
    n_rows = data.get('n_rows', 10)

    if not os.path.exists(file_path):
        return jsonify({'error': f'File not found: {file_path}'}), 404

    try:
        
        file_ext = os.path.splitext(file_path)[1].lower()
        
        file_size = os.path.getsize(file_path)
        
        
        if file_ext in ['.csv', '.txt']:
            
            df = pd.read_csv(file_path, nrows=n_rows)
            
            
            total_rows = sum(1 for _ in open(file_path, 'r')) - 1  
            
        elif file_ext in ['.xlsx', '.xls']:
            
            df = pd.read_excel(file_path, nrows=n_rows)
            
            
            import openpyxl
            wb = openpyxl.load_workbook(file_path, read_only=True)
            sheet = wb.active
            total_rows = sheet.max_row - 1  
            wb.close()
            
        else:
            return jsonify({'error': f'Unsupported file type: {file_ext}. Supported types: .csv, .txt, .xlsx, .xls'}), 400

        result = {
            'preview': df.to_dict(orient='records'),
            'columns': df.columns.tolist(),
            'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
            'metadata': {
                'file_name': os.path.basename(file_path),
                'file_path': file_path,
                'file_type': file_ext[1:],  
                'file_size': file_size,
                'file_size_formatted': format_size(file_size),
                'total_rows': total_rows,
                'preview_rows': len(df),
            }
        }

        return jsonify(result)

    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        return jsonify({'error': str(e), 'details': error_details}), 500


def format_size(size_bytes):
    """Format bytes into a human-readable string."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes/1024:.1f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes/(1024*1024):.1f} MB"
    else:
        return f"{size_bytes/(1024*1024*1024):.1f} GB"