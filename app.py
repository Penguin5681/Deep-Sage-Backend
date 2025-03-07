from kaggle.api.kaggle_api_extended import KaggleApi
import datetime
from flask import Flask, jsonify, request
import os
from flask_caching import Cache
from dotenv import load_dotenv

load_dotenv()

cache_config = {
    "CACHE_TYPE": "SimpleCache",
    "CACHE_DEFAULT_TIMEOUT": 300
}

app = Flask(__name__)

cache = Cache(app, config=cache_config)

os.environ['KAGGLE_USERNAME'] = os.getenv('KAGGLE_USERNAME')
os.environ['KAGGLE_KEY'] = os.getenv('KAGGLE_KEY')

kaggle_api = KaggleApi()


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
    """
    Retrieves a list of datasets from Kaggle and formats them into a standardized structure.

    This function fetches datasets from the Kaggle API based on the specified sorting criteria
    and limits the number of results. It validates the sorting option against allowed values
    and handles any exceptions that may occur during the API call.

    Args:
        limit (int, optional): Maximum number of datasets to return. Defaults to 5.
        sort_by (str, optional): Criteria to sort datasets by. Defaults to 'hottest'.
            Valid options are: 'hottest', 'votes', 'updated', 'active', 'published'.

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

            If an error occurs or sort_by is invalid, returns a dictionary with an 'error' key.

    Note:
        Requires prior authentication with the Kaggle API using authenticate_kaggle().
    """
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


@app.route('/api/datasets/kaggle', methods=['GET'])
@cache.memoize(300)
def get_kaggle_datasets_endpoint():
    """
    Flask endpoint that retrieves Kaggle datasets based on specified criteria.

    This endpoint requires Kaggle authentication credentials in the request headers.
    It extracts username and API key from headers, authenticates with the Kaggle API,
    and then calls the get_kaggle_datasets function with parameters from the request.

    HTTP Method: GET
    Route: /api/datasets/kaggle

    Request Headers:
        X-Kaggle-Username: Kaggle username for authentication
        X-Kaggle-Key: Kaggle API key for authentication

    Query Parameters:
        limit (int, optional): Maximum number of datasets to return. Defaults to 5.
        sort_by (str, optional): Criteria to sort datasets by. Defaults to 'hottest'.
            Valid options are: 'hottest', 'votes', 'updated', 'active', 'published'.

    Returns:
        JSON response containing either:
        - A list of formatted Kaggle datasets
        - An error message with appropriate HTTP status code (401 for authentication errors,
          500 for other errors)

    Note:
        Authentication is mandatory for this endpoint.
    """
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


@app.route('/api/datasets', methods=['GET'])
def get_datasets():
    """
    Flask endpoint that retrieves datasets from multiple sources based on specified parameters.

    This unified endpoint allows fetching datasets from both Kaggle and Hugging Face simultaneously
    or selectively, based on the 'source' parameter. It handles authentication for both platforms
    and applies source-specific sorting criteria.

    HTTP Method: GET
    Route: /api/datasets

    Request Headers:
        X-Kaggle-Username (optional): Kaggle username for authentication
        X-Kaggle-Key (optional): Kaggle API key for authentication
        X-HF-Token (optional): Hugging Face API token for authentication

    Query Parameters:
        source (str, optional): Source to fetch datasets from. Defaults to 'all'.
            Valid options are: 'all', 'kaggle', 'huggingface'
        limit (int, optional): Maximum number of datasets to return per source. Defaults to 5.
        kaggle_sort (str, optional): Criteria to sort Kaggle datasets by. Defaults to 'hottest'.
        hf_sort (str, optional): Criteria to sort Hugging Face datasets by. Defaults to 'downloads'.

    Returns:
        JSON response containing:
        - A dictionary with keys for each requested source ('kaggle', 'huggingface')
        - Each source key contains either a list of datasets or an error message
        - For Kaggle, returns an error if credentials are not provided

    Note:
        Authentication is required for Kaggle datasets, optional for Hugging Face.
    """
    source = request.args.get('source', 'all')
    limit = int(request.args.get('limit', 5))

    username = request.headers.get('X-Kaggle-Username')
    key = request.headers.get('X-Kaggle-Key')

    result = {}

    if source.lower() in ['all', 'kaggle']:
        if username and key:
            authenticate_kaggle(username, key)
            sort_by = request.args.get('kaggle_sort', 'hottest')
            result['kaggle'] = get_kaggle_datasets(limit, sort_by)
        else:
            result['kaggle'] = {'error': 'Kaggle credentials required'}

    return jsonify(result)


@app.route('/api/datasets/download', methods=['POST'])
def download_dataset():
    """
    Download dataset to user-specified path from multiple sources.

    Supports downloading datasets from Kaggle and HuggingFace to any user-specified location
    on their system. When running in Docker, handles path mapping between container and host.

    HTTP Method: POST
    Route: /api/datasets/download

    Request JSON:
        source (str): Source platform - 'kaggle' or 'huggingface'
        dataset_id (str): Dataset identifier (e.g. 'username/dataset-slug' for Kaggle)
        path (str): Target directory to download files (absolute or relative)
        config (str, optional): Configuration name for HuggingFace datasets

    Returns:
        JSON response with download status and path information
    """
    data = request.json
    source = data.get('source')
    dataset_id = data.get('dataset_id')
    download_path = data.get('path', './downloads')
    config = data.get('config')

    if not source or not dataset_id:
        return jsonify({'error': 'Source and dataset_id are required'}), 400

    # Ensure base downloads directory exists (for Docker volume mapping)
    base_downloads_dir = '/app/downloads'
    if not os.path.exists(base_downloads_dir):
        try:
            os.makedirs(base_downloads_dir, exist_ok=True)
            print(f"Created base downloads directory: {base_downloads_dir}")
        except Exception as e:
            print(f"Error creating base downloads directory: {str(e)}")
            return jsonify({'error': f'Unable to create downloads directory: {str(e)}'}), 500

    print(f"Received download path: {download_path}")

    if '\\' in download_path or ':' in download_path:
        # Windows-style path, use a safe container path instead
        safe_name = dataset_id.replace('/', '_').replace('\\', '_')
        container_path = f"{base_downloads_dir}/{safe_name}"
        print(f"Using container path for Docker: {container_path}")

        original_path = download_path
        download_path = container_path
    else:
        # Linux-style or relative path
        if not os.path.isabs(download_path):
            download_path = os.path.join(base_downloads_dir, download_path)
        original_path = download_path

    try:
        # Create target directory and any parent directories
        os.makedirs(download_path, exist_ok=True)
        print(f"Created directory: {download_path}")

        # Verify write permissions
        if not os.access(download_path, os.W_OK):
            return jsonify({'error': f'Directory {download_path} is not writable'}), 500

        # Process by source type
        if source.lower() == 'kaggle':
            username = request.headers.get('X-Kaggle-Username')
            key = request.headers.get('X-Kaggle-Key')

            if not username or not key:
                return jsonify({'error': 'Kaggle credentials required'}), 401

            authenticate_kaggle(username, key)
            print(
                f"Downloading Kaggle dataset {dataset_id} to {download_path}")

            kaggle_api.dataset_download_files(
                dataset_id,
                path=download_path,
                unzip=True
            )

            # List downloaded files
            files = os.listdir(download_path)
            print(f"Files in download directory: {files}")

            file_paths = [os.path.join(download_path, f) for f in files]

            return jsonify({
                'success': True,
                'message': f'Dataset downloaded successfully',
                'container_path': download_path,
                'host_path': original_path,
                'files': file_paths,
                'file_count': len(files),
                'docker_volume_path': f"./downloads/{os.path.basename(download_path)}"
            })
        else:
            return jsonify({
                'error': f'Invalid source "{source}". Supported sources: "kaggle", "huggingface"'
            }), 400

    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Download error: {str(e)}\n{error_details}")
        return jsonify({
            'error': str(e),
            'details': error_details
        }), 500


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

                result = {
                    'id': dataset_id,
                    'title': dataset['title'],
                    'owner': dataset['ownerName'],
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


@cache.memoize(timeout=1800)
def fetch_config_info(dataset_id, config):
    """
    Retrieves detailed configuration information for a specific Hugging Face dataset configuration.

    This function loads a dataset builder for the specified dataset ID and configuration,
    then extracts metadata including total size and number of examples. Results are cached
    for 30 minutes (1800 seconds) to improve performance and reduce API load.

    Args:
        dataset_id (str): The Hugging Face dataset identifier (e.g., 'squad', 'glue')
        config (str): The specific configuration name to fetch information for

    Returns:
        dict: A dictionary containing configuration information with the following keys:
            - name: Configuration name
            - total_size_bytes: Total size of the dataset configuration in bytes
            - total_examples: Total number of examples across all splits

            If an error occurs, the dictionary will contain:
            - name: Configuration name
            - error: Error message describing what went wrong

    Note:
        - Results are cached for 1800 seconds (30 minutes) to improve performance
        - This function is primarily used internally by search endpoints that need
          detailed configuration information
    """
    try:
        from datasets import load_dataset_builder
        builder = load_dataset_builder(dataset_id, config)
        info = builder.info

        total_size = 0
        num_examples = 0

        if hasattr(info, 'splits') and info.splits:
            for split_name, split_info in info.splits.items():
                total_size += getattr(split_info, 'num_bytes', 0)
                num_examples += getattr(split_info, 'num_examples', 0)

        return {
            'name': config,
            'total_size_bytes': total_size,
            'total_examples': num_examples
        }
    except Exception as e:
        return {
            'name': config,
            'error': f"Couldn't fetch info: {str(e)}"
        }


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
    """
    Flask endpoint that provides dataset name suggestions based on a search query.

    This endpoint provides autocomplete/typeahead functionality for dataset names from
    either Kaggle, Hugging Face, or both sources. It handles authentication for both
    platforms and returns dataset titles or IDs that match the query string.

    HTTP Method: GET
    Route: /api/suggestions

    Request Headers:
        X-Kaggle-Username (optional): Kaggle username for authentication
        X-Kaggle-Key (optional): Kaggle API key for authentication
        X-HF-Token (optional): Hugging Face API token for authentication

    Query Parameters:
        query (str, required): Search term to find matching dataset names
        source (str, optional): Source to get suggestions from. Defaults to 'all'.
            Valid options are: 'all', 'kaggle', 'huggingface'
        limit (int, optional): Maximum number of suggestions to return per source. Defaults to 5.

    Returns:
        JSON response containing:
        - A dictionary with keys for each requested source ('kaggle', 'huggingface')
        - Each source key contains either a list of dataset names or an error message
        - Returns empty array if query length is less than 1 character
        - For Kaggle, returns an error if credentials are not provided

    Note:
        - Authentication is required for Kaggle suggestions, optional for Hugging Face
        - Results are cached for 5 minutes to improve performance
    """
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
    """
    Flask endpoint that provides a health check for the service.

    This endpoint returns basic information about the service status and can be used
    by monitoring tools to verify that the API is operational.

    HTTP Method: GET
    Route: /health

    Returns:
        JSON response containing:
        - status: 'ok' if the service is running properly
        - version: API version information
        - timestamp: Current server time

    Status Code:
        200: Service is healthy and operational
    """
    return jsonify({
        'status': 'ok',
        'version': '1.0.0',
        'timestamp': datetime.datetime.now().isoformat()
    })


if __name__ == '__main__':
    app.run(debug=False, port=5000, host='0.0.0.0')
