import base64
from kaggle.api.kaggle_api_extended import KaggleApi
import datetime
from flask import Flask, jsonify, request, send_from_directory
import os
from flask_caching import Cache
from dotenv import load_dotenv
import threading
from data_processor import data_api
from data_processing.excel_to_csv import convert_api
from data_visualizations.pie_chart import pie_chart_bp
from ollama.ollama_insights import insights_api
from data_processing.csv_insights import csv_insights_api
from data_processing_flask_api.fix_basics_tab_api import data_cleaning_api

load_dotenv()

cache_config = {
    "CACHE_TYPE": "SimpleCache",
    "CACHE_DEFAULT_TIMEOUT": 300
}

app = Flask(__name__)
app.register_blueprint(data_api)
app.register_blueprint(convert_api)
app.register_blueprint(insights_api)
app.register_blueprint(csv_insights_api)
app.register_blueprint(data_cleaning_api)
app.register_blueprint(pie_chart_bp)


cache = Cache(app, config=cache_config)

os.environ['KAGGLE_USERNAME'] = os.getenv('KAGGLE_USERNAME')
os.environ['KAGGLE_KEY'] = os.getenv('KAGGLE_KEY')

kaggle_api = KaggleApi()

active_downloads = {}


def authenticate_kaggle(kaggle_username, kaggle_key):
    """
    Authenticates the Kaggle API using environment variables.
    """
    if not kaggle_username or not kaggle_key:
        raise ValueError(
            "KAGGLE_USERNAME and KAGGLE_KEY must be set in environment variables")

    kaggle_api.authenticate()
    return kaggle_api


def get_kaggle_datasets(limit=5, sort_by='hottest'):
    try:
        valid_sorts = ['hottest', 'votes', 'updated', 'active', 'published']
        if sort_by not in valid_sorts:
            return {'error': f'Invalid sort option. Choose from: {valid_sorts}'}

        datasets = list(kaggle_api.dataset_list(sort_by=sort_by))[:limit]
        formatted_datasets = []
        for dataset in datasets:
            formatted_datasets.append({
                'id': dataset.ref,
                'title': dataset.title,
                'owner': dataset.ownerName,
                'url': f'https://www.kaggle.com/datasets/{dataset.ref}',
                'size': dataset.size,
                'lastUpdated': dataset.lastUpdated,
                'downloadCount': dataset.downloadCount,
                'voteCount': dataset.voteCount,
                'description': dataset.description
            })

        return formatted_datasets
    except Exception as e:
        return {'error': str(e)}


@app.route('/static/pie_charts/<path:filename>')
def download_pie_chart(filename):
    return send_from_directory('/tmp/pie_charts', filename)


@app.route('/api/datasets/kaggle', methods=['GET'])
def get_kaggle_datasets_endpoint():

    username = request.headers.get('X-Kaggle-Username')
    key = request.headers.get('X-Kaggle-Key')

    if not username or not key:
        return jsonify({'error': 'Kaggle username and API key are required in headers'}), 401

    try:
        authenticate_kaggle(username, key)
        limit = int(request.args.get('limit', 5))
        sort_by = request.args.get('sort_by', 'hottest')
        result = get_kaggle_datasets(limit, sort_by)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/datasets/download', methods=['POST'])
def download_dataset():
    """
    Download dataset to user-specified path from multiple sources with real-time progress updates.

    Returns a stream of Server-Sent Events (SSE) with download progress that can be 
    consumed by a Flutter frontend to show progress indicators.
    """
    import platform
    import time
    import threading
    import json
    import queue
    import requests
    from pathlib import Path
    from flask import Response
    import copy

    data = request.json
    source = data.get('source')
    dataset_id = data.get('dataset_id')
    config = data.get('config')
    unzip = data.get('unzip', False)

    if platform.system() == 'Windows':
        default_download_path = os.path.join(
            os.environ['USERPROFILE'], 'Downloads')
    elif platform.system() == 'Darwin':
        default_download_path = os.path.join(os.environ['HOME'], 'Downloads')
    else:
        default_download_path = '/app/downloads'

    download_path = data.get('path', default_download_path)

    if not source or not dataset_id:
        return jsonify({'error': 'Source and dataset_id are required'}), 400

    kaggle_username = request.headers.get('X-Kaggle-Username')
    kaggle_key = request.headers.get('X-Kaggle-Key')

    if source.lower() == 'kaggle' and (not kaggle_username or not kaggle_key):
        return jsonify({'error': 'Kaggle credentials required'}), 401

    if source.lower() == 'kaggle':
        try:
            authenticate_kaggle(kaggle_username, kaggle_key)
        except Exception as e:
            return jsonify({'error': f'Kaggle authentication failed: {str(e)}'}), 401

    base_downloads_dir = '/app/downloads'
    is_docker = os.path.exists('/.dockerenv')

    if is_docker and not os.path.exists(base_downloads_dir):
        try:
            os.makedirs(base_downloads_dir, exist_ok=True)
            os.chmod(base_downloads_dir, 0o777)
        except Exception as e:
            return jsonify({'error': f'Unable to create downloads directory: {str(e)}'}), 500

    safe_name = dataset_id.replace('/', '_').replace('\\', '_')

    cancel_event = threading.Event()

    active_downloads[safe_name] = {
        'dataset_id': dataset_id,
        'cancel_event': cancel_event,
        'start_time': time.time()
    }

    status_queue = queue.Queue()

    def event_stream():
        """Generate SSE events with download progress updates."""
        download_status = {
            'state': 'initializing',
            'progress': 0,
            'message': 'Preparing download...',
            'dataset_id': dataset_id,
            'download_path': None,
            'files': [],
            'error': None
        }

        yield f"data: {json.dumps(download_status)}\n\n"

        try:
            if not os.path.exists(download_path):
                os.makedirs(download_path, exist_ok=True)
                download_status['message'] = f"Created directory: {download_path}"
                yield f"data: {json.dumps(download_status)}\n\n"

            dataset_dir = os.path.join(download_path, safe_name)
            os.makedirs(dataset_dir, exist_ok=True)

            final_download_path = dataset_dir
            original_path = dataset_dir

            if is_docker:
                container_path = f"{base_downloads_dir}/{safe_name}"
                container_path = container_path.replace('\\', '/')
                final_download_path = container_path

            try:
                os.chmod(final_download_path, 0o777)
            except Exception as e:
                download_status[
                    'message'] = f"Warning: Could not set permissions on {final_download_path}"
                yield f"data: {json.dumps(download_status)}\n\n"

            if not os.access(final_download_path, os.W_OK):
                download_status['state'] = 'error'
                download_status['error'] = f'Directory {final_download_path} is not writable'
                yield f"data: {json.dumps(download_status)}\n\n"
                return

            if source.lower() == 'kaggle':
                download_status['state'] = 'downloading'
                download_status['message'] = f"Starting download of {dataset_id}"
                download_status['download_path'] = final_download_path
                yield f"data: {json.dumps(download_status)}\n\n"

                download_completed = threading.Event()

                def monitor_download():
                    """Monitor download directory for changes to estimate progress."""
                    try:
                        initial_size = 0
                        if os.path.exists(final_download_path):
                            initial_size = sum(f.stat().st_size for f in Path(
                                final_download_path).glob('**/*') if f.is_file())

                        check_interval = 0.5
                        last_size = initial_size
                        last_update_time = time.time()
                        last_progress_reported = -1
                        start_time = time.time()

                        speeds = []
                        max_speed_samples = 10
                        estimated_total_size = None
                        stalled_count = 0
                        max_stalled_allowed = 40

                        while not download_completed.is_set() and not cancel_event.is_set():
                            time.sleep(check_interval)

                            if cancel_event.is_set():
                                status_queue.put({
                                    'state': 'cancelled',
                                    'progress': 0,
                                    'message': 'Download cancelled by user :('
                                })
                                break

                            try:
                                current_size = 0
                                if os.path.exists(final_download_path):
                                    current_size = sum(f.stat().st_size for f in Path(
                                        final_download_path).glob('**/*') if f.is_file())

                                current_time = time.time()
                                time_diff = current_time - last_update_time

                                if time_diff > 0:
                                    current_speed = (
                                        current_size - last_size) / time_diff
                                    if current_speed > 0:
                                        speeds.append(current_speed)
                                        if len(speeds) > max_speed_samples:
                                            speeds.pop(0)

                                        avg_speed = sum(
                                            speeds) / len(speeds) if speeds else 0

                                        if current_size == last_size:
                                            stalled_count += 1
                                        else:
                                            stalled_count = 0

                                        if not estimated_total_size or stalled_count > max_stalled_allowed:
                                            time_elapsed = current_time - start_time
                                            if time_elapsed > 5:
                                                new_estimate = current_size + \
                                                    (avg_speed * 30)
                                                if estimated_total_size is None:
                                                    estimated_total_size = new_estimate
                                                else:
                                                    estimated_total_size = max(
                                                        estimated_total_size, new_estimate)
                                                stalled_count = 0

                                        if estimated_total_size and estimated_total_size > initial_size:
                                            denominator = estimated_total_size - initial_size
                                            progress = int(
                                                (current_size - initial_size) / denominator * 100)
                                            progress = min(progress, 99)
                                        else:
                                            progress = 0

                                        if progress != last_progress_reported:
                                            status = {
                                                'state': 'downloading',
                                                'progress': progress,
                                                'message': f"Downloading... {progress}% ({format_size(current_size)})",
                                                'bytes_downloaded': current_size,
                                                'speed': f"{format_size(avg_speed)}/s" if avg_speed > 0 else "0 B/s",
                                                'estimated_size': estimated_total_size,
                                                'time_elapsed': int(current_time - start_time)
                                            }
                                            status_queue.put(status)

                                last_size = current_size
                                last_update_time = current_time
                            except Exception as e:
                                pass

                    except Exception as e:
                        status_queue.put({
                            'state': 'error',
                            'error': f"Monitor error: {str(e)}",
                            'message': f"Monitor error: {str(e)}"
                        })

                monitor_thread = threading.Thread(target=monitor_download)
                monitor_thread.daemon = True
                monitor_thread.start()

                download_thread_error = [None]

                download_request = [None]

                def perform_download(username, key, dataset):
                    """Perform download in separate thread using direct requests instead of Kaggle API."""
                    try:
                        if cancel_event.is_set():
                            status_queue.put({
                                'state': 'cancelled',
                                'message': 'Download cancelled before it started'
                            })
                            return

                        import kaggle
                        from kaggle.api.kaggle_api_extended import KaggleApi

                        os.environ['KAGGLE_USERNAME'] = username
                        os.environ['KAGGLE_KEY'] = key

                        api = KaggleApi()
                        api.authenticate()

                        dataset_owner, dataset_name = dataset.split('/')

                        auth_token = f"{username}:{key}"
                        encoded_token = base64.b64encode(
                            auth_token.encode()).decode()
                        headers = {
                            "Authorization": f"Basic {encoded_token}",
                            "User-Agent": "Kaggle/1.5.12"
                        }

                        download_url = f"https://www.kaggle.com/api/v1/datasets/download/{dataset}"

                        if cancel_event.is_set():
                            status_queue.put({
                                'state': 'cancelled',
                                'message': 'Download cancelled before started'
                            })
                            return

                        zip_path = os.path.join(
                            final_download_path, f"{dataset_name}.zip")

                        with requests.get(download_url, headers=headers, stream=True) as req:

                            download_request[0] = req

                            if req.status_code != 200:
                                raise Exception(
                                    f"Download failed with status code {req.status_code}: {req.text}")

                            total_size = int(
                                req.headers.get('content-length', 0))
                            if total_size > 0:
                                status_queue.put({
                                    'state': 'downloading',
                                    'message': f"Starting download of {format_size(total_size)}",
                                    'total_size': total_size
                                })

                            with open(zip_path, 'wb') as f:
                                for chunk in req.iter_content(chunk_size=8192):
                                    if cancel_event.is_set():

                                        req.close()
                                        download_completed.set()
                                        status_queue.put({
                                            'state': 'cancelled',
                                            'message': 'Download cancelled by user'
                                        })
                                        return

                                    if chunk:
                                        f.write(chunk)

                        if unzip and zip_path.endswith('.zip') and os.path.exists(zip_path):
                            import zipfile
                            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                                zip_ref.extractall(final_download_path)

                            if os.path.exists(zip_path):
                                os.remove(zip_path)

                        files = []
                        for root, _, filenames in os.walk(final_download_path):
                            for filename in filenames:
                                full_path = os.path.join(
                                    root, filename).replace('\\', '/')
                                host_path = full_path
                                if is_docker:
                                    host_path = full_path.replace(
                                        '/app/downloads', './downloads')

                                file_size = os.path.getsize(full_path)
                                files.append({
                                    'name': filename,
                                    'path': full_path,
                                    'size': file_size,
                                    'size_formatted': format_size(file_size),
                                    'container_path': full_path if is_docker else None,
                                    'host_path': host_path
                                })

                        host_dir_path = final_download_path
                        if is_docker:
                            host_dir_path = final_download_path.replace(
                                '/app/downloads', './downloads')

                        status_queue.put({
                            'state': 'completed',
                            'progress': 100,
                            'message': 'Dataset downloaded successfully',
                            'files': files,
                            'file_count': len(files),
                            'container_path': final_download_path if is_docker else None,
                            'host_path': original_path,
                            'host_volume_path': host_dir_path if is_docker else None,
                            'unzipped': unzip,
                            'safe_name': safe_name,
                            'docker_volume_path': host_dir_path if is_docker else None
                        })
                    except Exception as e:
                        download_thread_error[0] = str(e)
                        status_queue.put({
                            'state': 'error',
                            'error': str(e),
                            'message': f'Download failed: {str(e)}'
                        })
                    finally:
                        download_completed.set()
                        if safe_name in active_downloads:
                            del active_downloads[safe_name]

                download_thread = threading.Thread(
                    target=perform_download,
                    args=(kaggle_username, kaggle_key, dataset_id)
                )
                download_thread.daemon = True
                download_thread.start()

                while not download_completed.is_set() or not status_queue.empty():
                    try:
                        status_update = status_queue.get(timeout=0.5)

                        if status_update.get('state') == 'cancelled':
                            download_status.update(status_update)
                            yield f"data: {json.dumps(download_status)}\n\n"

                            try:
                                import shutil
                                if os.path.exists(final_download_path):
                                    shutil.rmtree(final_download_path)
                            except Exception as e:

                                print(
                                    f"Error cleaning up after cancellation: {str(e)}")

                            return

                        download_status.update(status_update)
                        yield f"data: {json.dumps(download_status)}\n\n"
                        status_queue.task_done()
                    except queue.Empty:
                        if cancel_event.is_set() and not download_completed.is_set():

                            if download_request[0] is not None:
                                try:
                                    download_request[0].close()
                                except:
                                    pass

                            download_status.update({
                                'state': 'cancelled',
                                'message': 'Download cancelled by user'
                            })
                            yield f"data: {json.dumps(download_status)}\n\n"

                            try:
                                import shutil
                                if os.path.exists(final_download_path):
                                    shutil.rmtree(final_download_path)
                            except:
                                pass

                            return

                if download_thread_error[0]:
                    download_status['state'] = 'error'
                    download_status['error'] = download_thread_error[0]
                    download_status['message'] = f'Download failed: {download_thread_error[0]}'
                    yield f"data: {json.dumps(download_status)}\n\n"
                elif download_status['state'] != 'completed':
                    download_status['state'] = 'completed'
                    download_status['progress'] = 100
                    download_status['message'] = 'Dataset downloaded successfully but no detailed results'
                    yield f"data: {json.dumps(download_status)}\n\n"

        except Exception as err:
            download_status['state'] = 'error'
            download_status['error'] = str(err)
            download_status['message'] = f'Download error: {str(err)}'
            yield f"data: {json.dumps(download_status)}\n\n"
        finally:
            if safe_name in active_downloads:
                del active_downloads[safe_name]

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

    return Response(event_stream(), mimetype="text/event-stream")


@app.route('/api/datasets/cancel', methods=['POST'])
def cancel_dataset_download():
    """
    Cancels an in-progress dataset download.

    HTTP Method: POST
    Route: /api/datasets/cancel

    Request Body:
        dataset_id: ID of the dataset being downloaded that should be cancelled

    Returns:
        JSON response indicating success or failure of the cancellation request
    """
    data = request.json
    dataset_id = data.get('dataset_id')

    if not dataset_id:
        return jsonify({'error': 'dataset_id is required'}), 400

    safe_key = dataset_id.replace('/', '_').replace('\\', '_')

    if safe_key not in active_downloads:
        return jsonify({
            'success': False,
            'message': f'No active download found for dataset: {dataset_id}'
        }), 404

    cancel_event = active_downloads[safe_key].get('cancel_event')

    if not cancel_event:
        return jsonify({
            'success': False,
            'message': 'Download exists but cannot be cancelled (no cancel event)'
        }), 500

    cancel_event.set()

    return jsonify({
        'success': True,
        'message': f'Cancellation signal sent for dataset: {dataset_id}',
        'dataset_id': dataset_id
    })


@cache.memoize(timeout=300)
def search_kaggle_datasets(query, limit=5):
    """
    Searches Kaggle datasets that match a query string and formats results.

    This function searches the Kaggle API for datasets matching the provided query
    and formats them into a standardized structure. Results are cached for 5 minutes
    to improve performance and reduce API calls.

    Args:
        query (str): Search query to find datasets
        limit (int, optional): Maximum number of datasets to return. Defaults to 5.

    Returns:
        list or dict: If successful, returns a list of dictionaries containing formatted dataset
            information with the following keys:
            - id: Dataset reference identifier
            - title: Dataset title
            - owner: Username of dataset owner
            - url: Full URL to the dataset on Kaggle
            - size: Size of the dataset
            - lastUpdated: When the dataset was last updated
            - downloadCount: Number of downloads
            - voteCount: Number of votes/upvotes
            - description: Dataset description

            If an error occurs, returns a dictionary with an 'error' key.

    Note:
        - Requires prior authentication with the Kaggle API using authenticate_kaggle().
        - Results are cached for 300 seconds (5 minutes) to improve performance.
    """
    try:
        datasets = list(kaggle_api.dataset_list(search=query))[:limit]
        formatted_datasets = []
        for dataset in datasets:
            formatted_datasets.append({
                'id': dataset.ref,
                'title': dataset.title,
                'owner': dataset.ownerName,
                'url': f'https://www.kaggle.com/datasets/{dataset.ref}',
                'size': dataset.size,
                'lastUpdated': dataset.lastUpdated,
                'downloadCount': dataset.downloadCount,
                'voteCount': dataset.voteCount,
                'description': dataset.description
            })
        return formatted_datasets
    except Exception as e:
        return {'error': str(e)}


@app.route('/api/search/kaggle', methods=['GET'])
def search_kaggle_endpoint():
    """
    Flask endpoint that searches for Kaggle datasets matching a query string.

    This endpoint requires Kaggle authentication credentials in the request headers.
    It extracts the search query from request parameters, authenticates with the Kaggle API,
    and then calls the search_kaggle_datasets function to perform the search.

    HTTP Method: GET
    Route: /api/search/kaggle

    Request Headers:
        X-Kaggle-Username: Kaggle username for authentication
        X-Kaggle-Key: Kaggle API key for authentication

    Query Parameters:
        query (str, required): Search term to find matching datasets
        limit (int, optional): Maximum number of datasets to return. Defaults to 5.

    Returns:
        JSON response containing either:
        - A list of formatted Kaggle datasets matching the search query
        - An error message with appropriate HTTP status code (401 for authentication errors,
          400 for missing query, 500 for other errors)

    Note:
        - Authentication is mandatory for this endpoint
        - Results are cached for 5 minutes to improve performance
    """
    username = request.headers.get('X-Kaggle-Username')
    key = request.headers.get('X-Kaggle-Key')
    query = request.args.get('query')

    if not username or not key:
        return jsonify({'error': 'Kaggle username and API key are required in headers'}), 401

    if not query:
        return jsonify({'error': 'Query parameter is required'}), 400

    try:
        authenticate_kaggle(username, key)
        limit = int(request.args.get('limit', 5))
        result = search_kaggle_datasets(query, limit)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/retrieve-kaggle-dataset', methods=["GET"])
def retrieve_kaggle_dataset():
    """
    Flask endpoint that retrieves metadata for a specific Kaggle dataset.

    This endpoint retrieves detailed metadata about a Kaggle dataset including its
    title, owner, size, last update time, download count, vote count, and description.
    It can handle both standard format 'username/dataset-slug' and plain dataset names.

    HTTP Method: GET
    Route: /api/retrieve-kaggle-dataset

    Request Headers:
        X-Kaggle-Username: Kaggle username for authentication
        X-Kaggle-Key: Kaggle API key for authentication

    Query Parameters:
        dataset_id (str, required): The Kaggle dataset identifier in format 'username/dataset-name'
                                    OR just the dataset name/title

    Returns:
        JSON response containing dataset metadata
    """
    username = request.headers.get('X-Kaggle-Username')
    key = request.headers.get('X-Kaggle-Key')
    dataset_id = request.args.get('dataset_id')

    if not dataset_id:
        return jsonify({'error': 'dataset_id parameter is required'}), 400

    if not username or not key:
        return jsonify({'error': 'Kaggle username and API key are required in headers'}), 401

    try:
        authenticate_kaggle(username, key)

        if '/' in dataset_id:
            parts = dataset_id.split('/')
            if len(parts) != 2:
                return jsonify({
                    'error': 'Invalid dataset_id format. Expected format: username/dataset-slug',
                    'example': 'netflix-inc/netflix-prize-data'
                }), 400

            owner, slug = parts

            try:
                dataset = kaggle_api.dataset_metadata(owner, slug)
                print("DATASET: ", dataset)
                result = {
                    'id': dataset_id,
                    'title': dataset['title'],
                    'owner': getattr(dataset, 'userName', getattr(dataset, 'owner', {}).get('username', 'Unknown')),
                    'url': f'https://www.kaggle.com/datasets/{dataset_id}',
                    'size': dataset.get('totalBytes', 'Unknown'),
                    'lastUpdated': dataset.get('lastUpdated', 'Unknown'),
                    'downloadCount': dataset.get('downloadCount', 0),
                    'voteCount': dataset.get('voteCount', 0),
                    'description': dataset.get('description', '')
                }

                return jsonify(result)
            except Exception:
                datasets = list(kaggle_api.dataset_list(
                    search=slug, user=owner))

                dataset = None
                for ds in datasets:
                    if ds.ref.lower() == dataset_id.lower():
                        dataset = ds
                        break

                if not dataset:
                    return jsonify({
                        'error': f'Dataset not found: {dataset_id}',
                        'suggestion': 'Try using just the dataset name or check the spelling'
                    }), 404
        else:
            datasets = list(kaggle_api.dataset_list(search=dataset_id))
            dataset = None
            for ds in datasets:
                if ds.title.lower() == dataset_id.lower():
                    dataset = ds
                    break

            if not dataset and datasets:
                dataset = datasets[0]

            if not dataset:
                return jsonify({
                    'error': f'No datasets found matching: {dataset_id}',
                    'suggestion': 'Try using more specific search terms'
                }), 404

        result = {
            'id': dataset.ref,
            'title': dataset.title,
            'owner': dataset.ownerName,
            'url': f'https://www.kaggle.com/datasets/{dataset.ref}',
            'size': dataset.size,
            'lastUpdated': dataset.lastUpdated,
            'downloadCount': dataset.downloadCount,
            'voteCount': dataset.voteCount,
            'description': dataset.description
        }

        return jsonify(result)

    except Exception as e:
        return jsonify({
            'error': str(e),
            'suggestion': 'Make sure your Kaggle credentials are correct'
        }), 500


@cache.memoize(timeout=600)
def get_finance_datasets(limit=10, sort_by='hottest'):
    """
    Retrieves finance-related datasets from Kaggle.

    Searches for datasets matching finance keywords and categories, returning
    the most relevant results based on the specified sorting method.

    Args:
        limit (int, optional): Maximum number of datasets to return. Defaults to 10.
        sort_by (str, optional): How to sort results. Defaults to 'hottest'.
            Options: 'hottest', 'votes', 'updated', 'active', 'published'

    Returns:
        list: Formatted finance-related datasets from Kaggle
    """
    try:

        search_terms = [
            "finance", "financial", "stock", "stocks", "trading",
            "investment", "banking", "economy", "economic", "cryptocurrency"
        ]

        all_results = []

        for term in search_terms:
            datasets = list(kaggle_api.dataset_list(
                search=term, sort_by=sort_by))
            for dataset in datasets:

                if not any(d.get('id') == dataset.ref for d in all_results):
                    all_results.append({
                        'id': dataset.ref,
                        'title': dataset.title,
                        'owner': dataset.ownerName,
                        'url': f'https://www.kaggle.com/datasets/{dataset.ref}',
                        'size': dataset.size,
                        'lastUpdated': dataset.lastUpdated,
                        'downloadCount': dataset.downloadCount,
                        'voteCount': dataset.voteCount,
                        'description': dataset.description,
                        'searchTerm': term
                    })

                if len(all_results) >= limit:
                    break

            if len(all_results) >= limit:
                break

        return all_results[:limit]

    except Exception as e:
        return {'error': str(e)}


@app.route('/api/datasets/finance', methods=['GET'])
def get_finance_datasets_endpoint():
    """
    Flask endpoint that returns finance-related datasets from Kaggle.

    Requires Kaggle authentication via request headers and returns datasets
    related to finance, investments, banking, stocks, etc.

    HTTP Method: GET
    Route: /api/datasets/finance

    Request Headers:
        X-Kaggle-Username: Kaggle username for authentication
        X-Kaggle-Key: Kaggle API key for authentication

    Query Parameters:
        limit (int, optional): Maximum number of datasets to return. Defaults to 10.
        sort_by (str, optional): How to sort results. Defaults to 'hottest'.

    Returns:
        JSON response containing a list of finance-related datasets
    """
    username = request.headers.get('X-Kaggle-Username')
    key = request.headers.get('X-Kaggle-Key')

    if not username or not key:
        return jsonify({'error': 'Kaggle username and API key are required in headers'}), 401

    try:
        authenticate_kaggle(username, key)
        limit = int(request.args.get('limit', 10))
        sort_by = request.args.get('sort_by', 'hottest')

        result = get_finance_datasets(limit, sort_by)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@cache.memoize(timeout=600)
def get_technology_datasets(limit=10, sort_by='hottest'):
    """
    Retrieves technology-related datasets from Kaggle.

    Searches for datasets matching technology keywords and categories, returning
    the most relevant results based on the specified sorting method.

    Args:
        limit (int, optional): Maximum number of datasets to return. Defaults to 10.
        sort_by (str, optional): How to sort results. Defaults to 'hottest'.
            Options: 'hottest', 'votes', 'updated', 'active', 'published'

    Returns:
        list: Formatted technology-related datasets from Kaggle
    """
    try:

        search_terms = [
            "technology", "tech", "software", "hardware", "AI",
            "artificial intelligence", "machine learning", "data science",
            "programming", "computer science", "robotics", "iot"
        ]

        all_results = []

        for term in search_terms:
            datasets = list(kaggle_api.dataset_list(
                search=term, sort_by=sort_by))
            for dataset in datasets:

                if not any(d.get('id') == dataset.ref for d in all_results):
                    all_results.append({
                        'id': dataset.ref,
                        'title': dataset.title,
                        'owner': dataset.ownerName,
                        'url': f'https://www.kaggle.com/datasets/{dataset.ref}',
                        'size': dataset.size,
                        'lastUpdated': dataset.lastUpdated,
                        'downloadCount': dataset.downloadCount,
                        'voteCount': dataset.voteCount,
                        'description': dataset.description,
                        'searchTerm': term
                    })

                if len(all_results) >= limit:
                    break

            if len(all_results) >= limit:
                break

        return all_results[:limit]

    except Exception as e:
        return {'error': str(e)}


@app.route('/api/datasets/technology', methods=['GET'])
def get_technology_datasets_endpoint():
    """
    Flask endpoint that returns technology-related datasets from Kaggle.

    Requires Kaggle authentication via request headers and returns datasets
    related to technology, AI, machine learning, software, hardware, etc.

    HTTP Method: GET
    Route: /api/datasets/technology

    Request Headers:
        X-Kaggle-Username: Kaggle username for authentication
        X-Kaggle-Key: Kaggle API key for authentication

    Query Parameters:
        limit (int, optional): Maximum number of datasets to return. Defaults to 10.
        sort_by (str, optional): How to sort results. Defaults to 'hottest'.

    Returns:
        JSON response containing a list of technology-related datasets
    """
    username = request.headers.get('X-Kaggle-Username')
    key = request.headers.get('X-Kaggle-Key')

    if not username or not key:
        return jsonify({'error': 'Kaggle username and API key are required in headers'}), 401

    try:
        authenticate_kaggle(username, key)
        limit = int(request.args.get('limit', 10))
        sort_by = request.args.get('sort_by', 'hottest')

        result = get_technology_datasets(limit, sort_by)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@cache.memoize(timeout=600)
def get_healthcare_datasets(limit=10, sort_by='hottest'):
    """
    Retrieves healthcare-related datasets from Kaggle.

    Searches for datasets matching healthcare keywords and categories, returning
    the most relevant results based on the specified sorting method.

    Args:
        limit (int, optional): Maximum number of datasets to return. Defaults to 10.
        sort_by (str, optional): How to sort results. Defaults to 'hottest'.
            Options: 'hottest', 'votes', 'updated', 'active', 'published'

    Returns:
        list: Formatted healthcare-related datasets from Kaggle
    """
    try:

        search_terms = [
            "healthcare", "health", "medical", "medicine", "clinical",
            "patient", "hospital", "disease", "diagnosis", "treatment",
            "pharmaceutical", "biomedical", "drug", "epidemiology", "covid", "diabetes", "cancer"
        ]

        all_results = []

        for term in search_terms:
            datasets = list(kaggle_api.dataset_list(
                search=term, sort_by=sort_by))
            for dataset in datasets:

                if not any(d.get('id') == dataset.ref for d in all_results):
                    all_results.append({
                        'id': dataset.ref,
                        'title': dataset.title,
                        'owner': dataset.ownerName,
                        'url': f'https://www.kaggle.com/datasets/{dataset.ref}',
                        'size': dataset.size,
                        'lastUpdated': dataset.lastUpdated,
                        'downloadCount': dataset.downloadCount,
                        'voteCount': dataset.voteCount,
                        'description': dataset.description,
                        'searchTerm': term
                    })

                if len(all_results) >= limit:
                    break

            if len(all_results) >= limit:
                break

        return all_results[:limit]

    except Exception as e:
        return {'error': str(e)}


@app.route('/api/datasets/healthcare', methods=['GET'])
def get_healthcare_datasets_endpoint():
    """
    Flask endpoint that returns healthcare-related datasets from Kaggle.

    Requires Kaggle authentication via request headers and returns datasets
    related to healthcare, medicine, clinical research, diseases, treatments, etc.

    HTTP Method: GET
    Route: /api/datasets/healthcare

    Request Headers:
        X-Kaggle-Username: Kaggle username for authentication
        X-Kaggle-Key: Kaggle API key for authentication

    Query Parameters:
        limit (int, optional): Maximum number of datasets to return. Defaults to 10.
        sort_by (str, optional): How to sort results. Defaults to 'hottest'.

    Returns:
        JSON response containing a list of healthcare-related datasets
    """
    username = request.headers.get('X-Kaggle-Username')
    key = request.headers.get('X-Kaggle-Key')

    if not username or not key:
        return jsonify({'error': 'Kaggle username and API key are required in headers'}), 401

    try:
        authenticate_kaggle(username, key)
        limit = int(request.args.get('limit', 10))
        sort_by = request.args.get('sort_by', 'hottest')

        result = get_healthcare_datasets(limit, sort_by)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/search', methods=['GET'])
def search_all_datasets():
    """
    Flask endpoint that searches for datasets across multiple sources based on a query string.

    This unified search endpoint allows searching for datasets on both Kaggle and Hugging Face 
    simultaneously or selectively, based on the 'source' parameter. It handles authentication 
    for both platforms and returns results from each requested source.

    HTTP Method: GET
    Route: /api/search

    Request Headers:
        X-Kaggle-Username (optional): Kaggle username for authentication
        X-Kaggle-Key (optional): Kaggle API key for authentication
        X-HF-Token (optional): Hugging Face API token for accessing private datasets

    Query Parameters:
        query (str, required): Search term to find matching datasets
        source (str, optional): Source to search in. Defaults to 'all'.
            Valid options are: 'all', 'kaggle', 'huggingface'
        limit (int, optional): Maximum number of datasets to return per source. Defaults to 5.

    Returns:
        JSON response containing:
        - A dictionary with keys for each requested source ('kaggle', 'huggingface')
        - Each source key contains either a list of matching datasets or an error message
        - For Kaggle, returns an error if credentials are not provided

    Status Codes:
        200: Success
        400: Missing query parameter

    Note:
        - Authentication is required for Kaggle search, optional for Hugging Face
        - Results utilize the cached search functions for better performance
    """
    query = request.args.get('query')
    if not query:
        return jsonify({'error': 'Query parameter is required'}), 400

    source = request.args.get('source', 'all')
    limit = int(request.args.get('limit', 5))

    username = request.headers.get('X-Kaggle-Username')
    key = request.headers.get('X-Kaggle-Key')
    token = request.headers.get('X-HF-Token')

    result = {}

    if source.lower() in ['all', 'kaggle']:
        if username and key:
            authenticate_kaggle(username, key)
            result['kaggle'] = search_kaggle_datasets(query, limit)
        else:
            result['kaggle'] = {'error': 'Kaggle credentials required'}
    return jsonify(result)


@cache.memoize(timeout=300)
def get_dataset_suggestions(query, source, limit=5):
    """
    Retrieves dataset name suggestions based on a search query from a specified source.

    This function fetches dataset titles or IDs that match the provided query from either
    Kaggle or Hugging Face. Results are cached for 5 minutes to improve performance and
    reduce API calls.

    Args:
        query (str): Search query to find matching datasets
        source (str): Source to search for suggestions, either 'kaggle' or 'huggingface'
        limit (int, optional): Maximum number of suggestions to return. Defaults to 5.

    Returns:
        list or dict: If successful, returns a list of dataset names/titles that match
            the search query. For Kaggle, returns dataset titles. For Hugging Face,
            returns dataset IDs.

            If an error occurs, returns a dictionary with an 'error' key.

    Note:
        - For Kaggle source, requires prior authentication with the Kaggle API
        - Results are cached for 300 seconds (5 minutes) to improve performance
        - This function is primarily used for autocomplete/typeahead functionality
    """
    suggestions = []

    try:
        if source.lower() == 'kaggle':
            datasets = list(kaggle_api.dataset_list(search=query))[:limit]
            suggestions = [dataset.title for dataset in datasets]
    except Exception as e:
        return {'error': str(e)}

    return suggestions


@app.route('/api/suggestions', methods=['GET'])
def get_suggestions():
    query = request.args.get('query', '')
    source = request.args.get('source', 'all')
    limit = int(request.args.get('limit', 5))

    if len(query) < 1:
        return jsonify([])

    result = {}

    if source.lower() in ['kaggle']:
        username = request.headers.get('X-Kaggle-Username')
        key = request.headers.get('X-Kaggle-Key')
        if username and key:
            authenticate_kaggle(username, key)
            result['kaggle'] = get_dataset_suggestions(query, 'kaggle', limit)
        else:
            result['kaggle'] = {'error': 'Kaggle credentials required'}

    return jsonify(result)


@app.route('/health', methods=['GET'])
def health_check():

    return jsonify({
        'status': 'ok',
        'version': '1.0.0',
        'timestamp': datetime.datetime.now().isoformat()
    })


@app.route('/api/docs', methods=['GET'])
def api_documentation():
    """
    Returns documentation for all API endpoints.

    This endpoint provides a comprehensive overview of all available API endpoints,
    their HTTP methods, required parameters, headers, and descriptions.

    Returns:
        JSON response containing documentation for all API endpoints
    """
    api_docs = {
        "version": "1.0.0",
        "baseUrl": request.url_root,
        "endpoints": [
            {
                "path": "/static/pie_charts/<filename>",
                "method": "GET",
                "description": "Retrieves a generated pie chart image file",
                "parameters": [
                    {
                        "name": "filename",
                        "in": "path",
                        "description": "Name of the pie chart image file",
                        "required": True
                    }
                ],
                "responses": {
                    "200": "Returns the requested pie chart image",
                    "404": "Chart not found"
                }
            },
            {
                "path": "/api/datasets/kaggle",
                "method": "GET",
                "description": "Retrieves a list of popular datasets from Kaggle",
                "headers": [
                    {
                        "name": "X-Kaggle-Username",
                        "description": "Kaggle username for authentication",
                        "required": True
                    },
                    {
                        "name": "X-Kaggle-Key",
                        "description": "Kaggle API key for authentication",
                        "required": True
                    }
                ],
                "parameters": [
                    {
                        "name": "limit",
                        "in": "query",
                        "description": "Maximum number of datasets to return",
                        "required": False,
                        "default": 5
                    },
                    {
                        "name": "sort_by",
                        "in": "query",
                        "description": "Sort datasets by this criteria",
                        "required": False,
                        "default": "hottest",
                        "options": ["hottest", "votes", "updated", "active", "published"]
                    }
                ],
                "responses": {
                    "200": "List of datasets",
                    "401": "Authentication failed",
                    "500": "Server error"
                }
            },
            {
                "path": "/api/datasets/download",
                "method": "POST",
                "description": "Downloads a dataset from Kaggle with real-time progress updates via Server-Sent Events",
                "headers": [
                    {
                        "name": "X-Kaggle-Username",
                        "description": "Kaggle username for authentication",
                        "required": True
                    },
                    {
                        "name": "X-Kaggle-Key",
                        "description": "Kaggle API key for authentication",
                        "required": True
                    }
                ],
                "requestBody": {
                    "content": "application/json",
                    "schema": {
                        "source": "Source of the dataset (e.g., 'kaggle')",
                        "dataset_id": "ID of the dataset to download",
                        "path": "Optional download path",
                        "unzip": "Whether to unzip the downloaded file (boolean)"
                    }
                },
                "responses": {
                    "200": "Stream of Server-Sent Events with download progress",
                    "400": "Missing required parameters",
                    "401": "Authentication failed",
                    "500": "Server error"
                }
            },
            {
                "path": "/api/datasets/cancel",
                "method": "POST",
                "description": "Cancels an in-progress dataset download",
                "requestBody": {
                    "content": "application/json",
                    "schema": {
                        "dataset_id": "ID of the dataset being downloaded that should be cancelled"
                    }
                },
                "responses": {
                    "200": "Cancellation successful",
                    "400": "Missing dataset_id",
                    "404": "No active download found",
                    "500": "Server error"
                }
            },
            {
                "path": "/api/search/kaggle",
                "method": "GET",
                "description": "Searches for Kaggle datasets matching a query string",
                "headers": [
                    {
                        "name": "X-Kaggle-Username",
                        "description": "Kaggle username for authentication",
                        "required": True
                    },
                    {
                        "name": "X-Kaggle-Key",
                        "description": "Kaggle API key for authentication",
                        "required": True
                    }
                ],
                "parameters": [
                    {
                        "name": "query",
                        "in": "query",
                        "description": "Search term to find matching datasets",
                        "required": True
                    },
                    {
                        "name": "limit",
                        "in": "query",
                        "description": "Maximum number of datasets to return",
                        "required": False,
                        "default": 5
                    }
                ],
                "responses": {
                    "200": "List of datasets matching the query",
                    "400": "Missing query parameter",
                    "401": "Authentication failed",
                    "500": "Server error"
                }
            },
            {
                "path": "/api/retrieve-kaggle-dataset",
                "method": "GET",
                "description": "Retrieves metadata for a specific Kaggle dataset",
                "headers": [
                    {
                        "name": "X-Kaggle-Username",
                        "description": "Kaggle username for authentication",
                        "required": True
                    },
                    {
                        "name": "X-Kaggle-Key",
                        "description": "Kaggle API key for authentication",
                        "required": True
                    }
                ],
                "parameters": [
                    {
                        "name": "dataset_id",
                        "in": "query",
                        "description": "The Kaggle dataset identifier (username/dataset-name or dataset name)",
                        "required": True
                    }
                ],
                "responses": {
                    "200": "Dataset metadata",
                    "400": "Missing dataset_id parameter",
                    "401": "Authentication failed",
                    "404": "Dataset not found",
                    "500": "Server error"
                }
            },
            {
                "path": "/api/datasets/finance",
                "method": "GET",
                "description": "Retrieves finance-related datasets from Kaggle",
                "headers": [
                    {
                        "name": "X-Kaggle-Username",
                        "description": "Kaggle username for authentication",
                        "required": True
                    },
                    {
                        "name": "X-Kaggle-Key",
                        "description": "Kaggle API key for authentication",
                        "required": True
                    }
                ],
                "parameters": [
                    {
                        "name": "limit",
                        "in": "query",
                        "description": "Maximum number of datasets to return",
                        "required": False,
                        "default": 10
                    },
                    {
                        "name": "sort_by",
                        "in": "query",
                        "description": "Sort datasets by this criteria",
                        "required": False,
                        "default": "hottest",
                        "options": ["hottest", "votes", "updated", "active", "published"]
                    }
                ],
                "responses": {
                    "200": "List of finance-related datasets",
                    "401": "Authentication failed",
                    "500": "Server error"
                }
            },
            {
                "path": "/api/datasets/technology",
                "method": "GET",
                "description": "Retrieves technology-related datasets from Kaggle",
                "headers": [
                    {
                        "name": "X-Kaggle-Username",
                        "description": "Kaggle username for authentication",
                        "required": True
                    },
                    {
                        "name": "X-Kaggle-Key",
                        "description": "Kaggle API key for authentication",
                        "required": True
                    }
                ],
                "parameters": [
                    {
                        "name": "limit",
                        "in": "query",
                        "description": "Maximum number of datasets to return",
                        "required": False,
                        "default": 10
                    },
                    {
                        "name": "sort_by",
                        "in": "query",
                        "description": "Sort datasets by this criteria",
                        "required": False,
                        "default": "hottest",
                        "options": ["hottest", "votes", "updated", "active", "published"]
                    }
                ],
                "responses": {
                    "200": "List of technology-related datasets",
                    "401": "Authentication failed",
                    "500": "Server error"
                }
            },
            {
                "path": "/api/datasets/healthcare",
                "method": "GET",
                "description": "Retrieves healthcare-related datasets from Kaggle",
                "headers": [
                    {
                        "name": "X-Kaggle-Username",
                        "description": "Kaggle username for authentication",
                        "required": True
                    },
                    {
                        "name": "X-Kaggle-Key",
                        "description": "Kaggle API key for authentication",
                        "required": True
                    }
                ],
                "parameters": [
                    {
                        "name": "limit",
                        "in": "query",
                        "description": "Maximum number of datasets to return",
                        "required": False,
                        "default": 10
                    },
                    {
                        "name": "sort_by",
                        "in": "query",
                        "description": "Sort datasets by this criteria",
                        "required": False,
                        "default": "hottest",
                        "options": ["hottest", "votes", "updated", "active", "published"]
                    }
                ],
                "responses": {
                    "200": "List of healthcare-related datasets",
                    "401": "Authentication failed",
                    "500": "Server error"
                }
            },
            {
                "path": "/api/search",
                "method": "GET",
                "description": "Unified search endpoint for datasets across multiple sources",
                "headers": [
                    {
                        "name": "X-Kaggle-Username",
                        "description": "Kaggle username for authentication (required for Kaggle search)",
                        "required": False
                    },
                    {
                        "name": "X-Kaggle-Key",
                        "description": "Kaggle API key for authentication (required for Kaggle search)",
                        "required": False
                    }
                ],
                "parameters": [
                    {
                        "name": "query",
                        "in": "query",
                        "description": "Search term to find matching datasets",
                        "required": True
                    },
                    {
                        "name": "source",
                        "in": "query",
                        "description": "Source to search in",
                        "required": False,
                        "default": "all",
                        "options": ["all", "kaggle"]
                    },
                    {
                        "name": "limit",
                        "in": "query",
                        "description": "Maximum number of datasets to return per source",
                        "required": False,
                        "default": 5
                    }
                ],
                "responses": {
                    "200": "Dictionary with search results from each source",
                    "400": "Missing query parameter",
                    "500": "Server error"
                }
            },
            {
                "path": "/api/suggestions",
                "method": "GET",
                "description": "Get dataset name suggestions for autocomplete functionality",
                "headers": [
                    {
                        "name": "X-Kaggle-Username",
                        "description": "Kaggle username for authentication (required for Kaggle suggestions)",
                        "required": False
                    },
                    {
                        "name": "X-Kaggle-Key",
                        "description": "Kaggle API key for authentication (required for Kaggle suggestions)",
                        "required": False
                    }
                ],
                "parameters": [
                    {
                        "name": "query",
                        "in": "query",
                        "description": "Partial search term to find matching dataset names",
                        "required": True
                    },
                    {
                        "name": "source",
                        "in": "query",
                        "description": "Source to get suggestions from",
                        "required": False,
                        "default": "all",
                        "options": ["all", "kaggle"]
                    },
                    {
                        "name": "limit",
                        "in": "query",
                        "description": "Maximum number of suggestions to return",
                        "required": False,
                        "default": 5
                    }
                ],
                "responses": {
                    "200": "List of dataset title suggestions",
                    "500": "Server error"
                }
            },
            {
                "path": "/health",
                "method": "GET",
                "description": "Health check endpoint to verify API is operational",
                "responses": {
                    "200": "API is operational with status information"
                }
            },
            {
                "path": "/api/docs",
                "method": "GET",
                "description": "This documentation endpoint",
                "responses": {
                    "200": "API documentation"
                }
            }
        ]
    }

    return jsonify(api_docs)


if __name__ == '__main__':
    app.run(debug=True, port=5000, host='0.0.0.0')
