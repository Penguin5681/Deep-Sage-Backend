import pandas as pd
import os
from flask import Blueprint, jsonify, request
import time
import threading
import queue
from pathlib import Path

convert_api = Blueprint('convert_api', __name__)

active_conversions = {}


@convert_api.route('/api/convert/excel-to-csv', methods=['POST'])
def excel_to_csv():
    """
    Convert Excel files to CSV format efficiently.

    Request JSON:
    {
        "file_path": "/path/to/file.xlsx",       
        "output_dir": "/path/to/output/dir",     
        "sheet_name": "Sheet1",                  
        "encoding": "utf-8",                     
        "delimiter": ",",                        
        "date_format": None,                     
        "decimal": ".",                          
        "include_index": false,                  
        "keep_original_name": true               
    }

    Returns JSON:
    {
        "success": true,
        "csv_paths": ["/path/to/output.csv", ...],
        "execution_time": "2.5s",
        "sheets_converted": 3
    }
    """
    data = request.json

    if not data or 'file_path' not in data:
        return jsonify({'error': 'file_path is required'}), 400

    file_path = data['file_path']

    if not os.path.exists(file_path):
        return jsonify({'error': f'File not found: {file_path}'}), 404

    file_ext = os.path.splitext(file_path)[1].lower()
    if file_ext not in ['.xlsx', '.xls']:
        return jsonify({'error': f'File is not an Excel file: {file_ext}'}), 400

    default_output_dir = os.path.dirname(file_path)
    output_dir = data.get('output_dir', default_output_dir)

    sheet_name = data.get('sheet_name', None)
    encoding = data.get('encoding', 'utf-8')
    delimiter = data.get('delimiter', ',')
    date_format = data.get('date_format', None)
    decimal = data.get('decimal', '.')
    include_index = data.get('include_index', False)

    keep_original_name = data.get('keep_original_name', True)

    os.makedirs(output_dir, exist_ok=True)

    try:
        start_time = time.time()

        excel_file = pd.ExcelFile(file_path)

        if sheet_name:
            if sheet_name in excel_file.sheet_names:
                sheets_to_process = [sheet_name]
            else:
                return jsonify({'error': f'Sheet "{sheet_name}" not found in the Excel file'}), 400
        else:

            sheets_to_process = [excel_file.sheet_names[0]]

        csv_paths = []
        base_name = os.path.splitext(os.path.basename(file_path))[0]

        for sheet in sheets_to_process:

            if keep_original_name and len(sheets_to_process) == 1:
                output_file = os.path.join(output_dir, f"{base_name}.csv")
            else:

                safe_sheet_name = "".join(
                    [c if c.isalnum() else "_" for c in sheet])
                output_file = os.path.join(
                    output_dir, f"{base_name}_{safe_sheet_name}.csv")

            df = pd.read_excel(
                file_path,
                sheet_name=sheet,
                engine='openpyxl' if file_ext == '.xlsx' else 'xlrd'
            )

            df.to_csv(
                output_file,
                encoding=encoding,
                sep=delimiter,
                date_format=date_format,
                decimal=decimal,
                index=include_index
            )

            csv_paths.append(output_file)

        execution_time = time.time() - start_time

        return jsonify({
            'success': True,
            'csv_paths': csv_paths,
            'execution_time': f"{execution_time:.2f}s",
            'sheets_converted': len(sheets_to_process)
        })

    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        return jsonify({'error': str(e), 'details': error_details}), 500


@convert_api.route('/api/convert/excel-to-csv/async', methods=['POST'])
def excel_to_csv_async():
    """
    Asynchronously convert Excel files to CSV with progress tracking.

    Returns a conversion ID that can be used to check status.
    """
    data = request.json

    if not data or 'file_path' not in data:
        return jsonify({'error': 'file_path is required'}), 400

    file_path = data['file_path']

    if not os.path.exists(file_path):
        return jsonify({'error': f'File not found: {file_path}'}), 404

    conversion_id = f"conv_{int(time.time())}_{os.path.basename(file_path)}"

    active_conversions[conversion_id] = {
        'file_path': file_path,
        'status': 'starting',
        'progress': 0,
        'start_time': time.time(),
        'csv_paths': [],
        'error': None,
        'queue': queue.Queue()
    }

    conversion_thread = threading.Thread(
        target=process_excel_async,
        args=(conversion_id, data)
    )
    conversion_thread.daemon = True
    conversion_thread.start()

    return jsonify({
        'conversion_id': conversion_id,
        'status': 'started',
        'message': 'Conversion started'
    })


@convert_api.route('/api/convert/excel-to-csv/status/<conversion_id>', methods=['GET'])
def check_conversion_status(conversion_id):
    """Check status of an asynchronous conversion."""
    if conversion_id not in active_conversions:
        return jsonify({'error': 'Conversion ID not found'}), 404

    conversion = active_conversions[conversion_id]

    response = {
        'status': conversion['status'],
        'progress': conversion['progress'],
        'file_path': conversion['file_path'],
        'elapsed_time': f"{time.time() - conversion['start_time']:.1f}s"
    }

    if conversion['status'] == 'completed':
        response['csv_paths'] = conversion['csv_paths']

        def cleanup():
            time.sleep(600)
            if conversion_id in active_conversions:
                del active_conversions[conversion_id]

        cleanup_thread = threading.Thread(target=cleanup)
        cleanup_thread.daemon = True
        cleanup_thread.start()

    elif conversion['status'] == 'error':
        response['error'] = conversion['error']

    return jsonify(response)


def process_excel_async(conversion_id, data):
    """Process Excel file conversion in a background thread."""
    conversion = active_conversions[conversion_id]

    try:
        file_path = data['file_path']
        output_dir = data.get('output_dir', os.path.dirname(file_path))
        sheet_name = data.get('sheet_name', None)
        encoding = data.get('encoding', 'utf-8')
        delimiter = data.get('delimiter', ',')
        date_format = data.get('date_format', None)
        decimal = data.get('decimal', '.')
        include_index = data.get('include_index', False)
        keep_original_name = data.get('keep_original_name', True)

        os.makedirs(output_dir, exist_ok=True)

        conversion['status'] = 'reading_excel'
        conversion['progress'] = 10

        file_ext = os.path.splitext(file_path)[1].lower()
        excel_file = pd.ExcelFile(file_path)

        if sheet_name:
            if sheet_name in excel_file.sheet_names:
                sheets_to_process = [sheet_name]
            else:
                raise ValueError(
                    f'Sheet "{sheet_name}" not found in the Excel file')
        else:

            sheets_to_process = [excel_file.sheet_names[0]]

        conversion['status'] = 'converting'
        conversion['progress'] = 20

        csv_paths = []
        total_sheets = len(sheets_to_process)
        base_name = os.path.splitext(os.path.basename(file_path))[0]

        for i, sheet in enumerate(sheets_to_process):
            sheet_progress = int(20 + (i / total_sheets) * 70)
            conversion['progress'] = sheet_progress
            conversion['status'] = f'converting_sheet_{i+1}_of_{total_sheets}'

            if keep_original_name and len(sheets_to_process) == 1:
                output_file = os.path.join(output_dir, f"{base_name}.csv")
            else:

                safe_sheet_name = "".join(
                    [c if c.isalnum() else "_" for c in sheet])
                output_file = os.path.join(
                    output_dir, f"{base_name}_{safe_sheet_name}.csv")

            df = pd.read_excel(
                file_path,
                sheet_name=sheet,
                engine='openpyxl' if file_ext == '.xlsx' else 'xlrd'
            )

            df.to_csv(
                output_file,
                encoding=encoding,
                sep=delimiter,
                date_format=date_format,
                decimal=decimal,
                index=include_index
            )

            csv_paths.append(output_file)

        conversion['status'] = 'completed'
        conversion['progress'] = 100
        conversion['csv_paths'] = csv_paths

    except Exception as e:
        import traceback
        conversion['status'] = 'error'
        conversion['error'] = str(e)
        conversion['details'] = traceback.format_exc()
        conversion['progress'] = 0
