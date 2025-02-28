from flask import Flask, jsonify, request
import os
import requests
from kaggle.api.kaggle_api_extended import KaggleApi
from huggingface_hub import HfApi
from flask_caching import Cache
from datasets import get_dataset_config_names
import concurrent.futures

cache_config = {
    "CACHE_TYPE": "SimpleCache",
    "CACHE_DEFAULT_TIMEOUT": 300  # The cached results will be valid for 5 mins
}

app = Flask(__name__)

cache = Cache(app, config=cache_config)

kaggle_api = KaggleApi()
hf_api = HfApi()


def authenticate_kaggle(username, key):
    """
    Authenticates the Kaggle API with the provided credentials.

    This function sets the necessary environment variables for Kaggle API authentication
    and initializes the API connection. It must be called before making any Kaggle API requests.

    Args:
        username (str): The Kaggle username to use for authentication
        key (str): The Kaggle API key associated with the username

    Returns:
        None

    Note:
        The credentials are stored as environment variables 'KAGGLE_USERNAME' and 'KAGGLE_KEY'
        The kaggle_api object must be initialized before calling this function
    """
    os.environ['KAGGLE_USERNAME'] = username
    os.environ['KAGGLE_KEY'] = key
    kaggle_api.authenticate()


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


def get_huggingface_datasets(limit=5, sort_by='downloads'):
    """
    Retrieves a list of datasets from Hugging Face and formats them into a standardized structure.
    
    This function fetches datasets from the Hugging Face API based on the specified sorting criteria
    and limits the number of results. It validates the sorting option against allowed values
    and handles any exceptions that may occur during the API call.
    
    Args:
        limit (int, optional): Maximum number of datasets to return. Defaults to 5.
        sort_by (str, optional): Criteria to sort datasets by. Defaults to 'downloads'.
            Valid options are: 'downloads', 'trending', 'modified'.
            
    Returns:
        list or dict: If successful, returns a list of dictionaries containing formatted dataset
            information with the following keys:
            - id: Dataset identifier
            - author: Username of dataset author
            - url: Full URL to the dataset on Hugging Face
            - downloads: Number of downloads
            - likes: Number of likes/upvotes
            - lastModified: When the dataset was last modified
            - tags: List of dataset tags
            - description: Dataset description
            
            If an error occurs or sort_by is invalid, returns a dictionary with an 'error' key.
    """
    try:
        valid_sorts = ['downloads', 'trending', 'modified']
        if sort_by not in valid_sorts:
            return {'error': f'Invalid sort option. Choose from: {valid_sorts}'}

        response = requests.get(
            'https://huggingface.co/api/datasets',
            params={'sort': sort_by, 'limit': limit}
        )

        if response.status_code == 200:
            datasets = response.json()
        else:
            return {'error': f'API returned status code {response.status_code}'}

        formatted_datasets = []
        for dataset in datasets:
            formatted_datasets.append({
                'id': dataset.get('id'),
                'author': dataset.get('author'),
                'url': f"https://huggingface.co/datasets/{dataset.get('id')}",
                'downloads': dataset.get('downloads'),
                'likes': dataset.get('likes'),
                'lastModified': dataset.get('lastModified'),
                'tags': dataset.get('tags', []),
                'description': dataset.get('description', '')
            })
        return formatted_datasets
    except Exception as e:
        return {'error': str(e)}


@app.route('/api/datasets/kaggle', methods=['GET'])
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


@app.route('/api/datasets/huggingface', methods=['GET'])
def get_huggingface_datasets_endpoint():
    """
    Flask endpoint that retrieves Hugging Face datasets based on specified criteria.
    
    This endpoint allows optional authentication with a Hugging Face token provided in the
    request headers. It extracts query parameters for limit and sorting criteria, then
    calls the get_huggingface_datasets function to fetch and format dataset information.
    
    HTTP Method: GET
    Route: /api/datasets/huggingface
    
    Request Headers:
        X-HF-Token (optional): Hugging Face API token for accessing private or gated datasets
        
    Query Parameters:
        limit (int, optional): Maximum number of datasets to return. Defaults to 5.
        sort_by (str, optional): Criteria to sort datasets by. Defaults to 'downloads'.
            Valid options are: 'downloads', 'trending', 'modified'.
            
    Returns:
        JSON response containing either:
        - A list of formatted Hugging Face datasets
        - An error message with appropriate HTTP status code
        
    Note:
        Unlike the Kaggle endpoint, authentication is optional for this endpoint.
    """
    token = request.headers.get('X-HF-Token')
    if token:
        os.environ['HF_TOKEN'] = token

    limit = int(request.args.get('limit', 5))
    sort_by = request.args.get('sort_by', 'downloads')
    result = get_huggingface_datasets(limit, sort_by)
    return jsonify(result)


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

    if source.lower() in ['all', 'huggingface']:
        token = request.headers.get('X-HF-Token')
        if token:
            os.environ['HF_TOKEN'] = token
        sort_by = request.args.get('hf_sort', 'downloads')
        result['huggingface'] = get_huggingface_datasets(limit, sort_by)

    return jsonify(result)


@app.route('/api/datasets/configs', methods=['GET'])
def get_dataset_configs():
    """
    Flask endpoint that retrieves configuration information for a specified Hugging Face dataset.
    
    This endpoint fetches all available configurations for a dataset and provides detailed 
    information about each configuration including size, number of examples, splits, and features.
    For each configuration, it attempts to load the dataset builder to extract metadata.
    
    HTTP Method: GET
    Route: /api/datasets/configs
    
    Query Parameters:
        dataset_id (str, required): The Hugging Face dataset identifier (e.g., 'squad', 'glue')
            
    Returns:
        JSON response containing:
        - dataset_id: The requested dataset identifier
        - configs: List of configuration objects, each containing:
          - name: Configuration name
          - total_size_bytes: Total size of the dataset in bytes
          - total_examples: Total number of examples across all splits
          - splits: Dictionary of splits with size and example count information
          - description: Dataset description
          - features: String representation of dataset features
          
        If a configuration cannot be loaded, its entry will include an error message.
        If the dataset_id parameter is missing or an error occurs, returns an appropriate error response.
        
    Status Codes:
        200: Success
        400: Missing dataset_id parameter
        500: Server error when retrieving configurations
    """
    dataset_id = request.args.get('dataset_id')
    if not dataset_id:
        return jsonify({'error': 'dataset_id parameter is required'}), 400

    try:
        configs = get_dataset_config_names(dataset_id)

        configs_with_size = []
        from datasets import load_dataset_builder

        for config in configs:
            try:
                builder = load_dataset_builder(dataset_id, config)

                info = builder.info

                total_size = 0
                num_examples = 0
                splits = {}

                if hasattr(info, 'splits') and info.splits:
                    for split_name, split_info in info.splits.items():
                        split_size = getattr(split_info, 'num_bytes', 0)
                        split_examples = getattr(split_info, 'num_examples', 0)
                        total_size += split_size
                        num_examples += split_examples
                        splits[split_name] = {
                            'size_bytes': split_size,
                            'num_examples': split_examples
                        }

                config_info = {
                    'name': config,
                    'total_size_bytes': total_size,
                    'total_examples': num_examples,
                    'splits': splits,
                    'description': getattr(info, 'description', ''),
                    'features': str(getattr(info, 'features', {}))
                }

                configs_with_size.append(config_info)

            except Exception as e:
                configs_with_size.append({
                    'name': config,
                    'error': f"Couldn't fetch size: {str(e)}"
                })

        return jsonify({
            'dataset_id': dataset_id,
            'configs': configs_with_size
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/datasets/download', methods=['POST'])
def download_dataset():
    """
    Flask endpoint that downloads a dataset from either Kaggle or Hugging Face based on the specified parameters.
    
    This endpoint handles downloading datasets from multiple sources, each with their own authentication
    requirements and download mechanisms. It supports specifying a download location and, for Hugging Face
    datasets, configuration options.
    
    HTTP Method: POST
    Route: /api/datasets/download
    
    Request Headers:
        X-Kaggle-Username (required for Kaggle): Kaggle username for authentication
        X-Kaggle-Key (required for Kaggle): Kaggle API key for authentication
        X-HF-Token (optional for Hugging Face): Hugging Face API token for accessing private datasets
        
    Request Body (JSON):
        source (str, required): Source to download from, either 'kaggle' or 'huggingface'
        dataset_id (str, required): Dataset identifier to download
        path (str, optional): Local path to save the dataset to. Defaults to './datasets'
        config (str, optional): Configuration name for Hugging Face datasets
            
    Returns:
        JSON response containing:
        - For successful downloads: success status and message with download details
        - For missing configs: error with available configurations for the dataset
        - For errors: appropriate error message
        
    Status Codes:
        200: Success
        400: Missing required parameters or invalid source
        401: Missing authentication credentials
        500: Error during download process
        
    Note:
        - Kaggle datasets are downloaded as zip files and not automatically extracted
        - Hugging Face datasets are converted to CSV format and saved to the specified path
    """
    data = request.json
    source = data.get('source')
    dataset_id = data.get('dataset_id')
    path = data.get('path', './datasets')
    config = data.get('config')

    username = request.headers.get('X-Kaggle-Username')
    key = request.headers.get('X-Kaggle-Key')
    token = request.headers.get('X-HF-Token')

    if not source or not dataset_id:
        return jsonify({'error': 'Source and dataset_id are required'}), 400

    try:
        if source.lower() == 'kaggle':
            if not username or not key:
                return jsonify({'error': 'Kaggle username and API key are required'}), 401

            authenticate_kaggle(username, key)
            kaggle_api.dataset_download_files(dataset_id, path, unzip=False)
            return jsonify({'success': True, 'message': f'Dataset {dataset_id} downloaded successfully to {path}'})

        elif source.lower() == 'huggingface':
            if token:
                os.environ['HF_TOKEN'] = token

            from datasets import load_dataset
            try:
                if config:
                    dataset = load_dataset(dataset_id, config, cache_dir=path)
                else:
                    dataset = load_dataset(dataset_id, cache_dir=path)
            except ValueError as e:
                if "Config name is missing" in str(e):
                    import re
                    configs = re.search(
                        r"pick one among the available configs: \[(.*?)\]", str(e))
                    if configs:
                        available_configs = configs.group(
                            1).replace("'", "").split(', ')
                        return jsonify({
                            'error': 'Config name is missing',
                            'available_configs': available_configs,
                            'message': f"Please specify a config from: {', '.join(available_configs)}."
                        }), 400
                raise e

            if isinstance(dataset, dict):
                for split_name, split_dataset in dataset.items():
                    split_dataset.to_pandas().to_csv(
                        f"{path}/{dataset_id.replace('/', '_')}_{split_name}.csv", index=False)
            else:
                dataset.to_pandas().to_csv(
                    f"{path}/{dataset_id.replace('/', '_')}.csv", index=False)

            config_info = f" with config '{config}'" if config else ""
            return jsonify({'success': True, 'message': f'Dataset {dataset_id}{config_info} downloaded successfully to {path}'})

        else:
            return jsonify({'error': 'Invalid source. Use "kaggle" or "huggingface"'}), 400

    except Exception as e:
        return jsonify({'error': f'Error downloading dataset: {str(e)}'}), 500


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


@cache.memoize(timeout=300)
def search_huggingface_datasets(query, limit=5):
    """
    Searches Hugging Face datasets that match a query string and formats results.
    
    This function searches the Hugging Face API for datasets matching the provided query
    and formats them into a standardized structure. Results are cached for 5 minutes
    to improve performance and reduce API calls.
    
    Args:
        query (str): Search query to find datasets
        limit (int, optional): Maximum number of datasets to return. Defaults to 5.
            
    Returns:
        list or dict: If successful, returns a list of dictionaries containing formatted dataset
            information with the following keys:
            - id: Dataset identifier
            - author: Username of dataset author
            - url: Full URL to the dataset on Hugging Face
            - downloads: Number of downloads
            - likes: Number of likes/upvotes
            - lastModified: When the dataset was last modified
            - tags: List of dataset tags
            - description: Dataset description
            
            If an error occurs, returns a dictionary with an 'error' key.
            
    Note:
        - Results are cached for 300 seconds (5 minutes) to improve performance.
        - Authentication is optional but may be necessary for accessing private datasets.
    """
    try:
        response = requests.get(
            'https://huggingface.co/api/datasets',
            params={'search': query, 'limit': limit}
        )

        if response.status_code == 200:
            datasets = response.json()
        else:
            return {'error': f'API returned status code {response.status_code}'}

        formatted_datasets = []
        for dataset in datasets:
            formatted_datasets.append({
                'id': dataset.get('id'),
                'author': dataset.get('author'),
                'url': f"https://huggingface.co/datasets/{dataset.get('id')}",
                'downloads': dataset.get('downloads'),
                'likes': dataset.get('likes'),
                'lastModified': dataset.get('lastModified'),
                'tags': dataset.get('tags', []),
                'description': dataset.get('description', '')
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


@app.route('/api/search/huggingface', methods=['GET'])
def search_huggingface_endpoint():
    """
    Flask endpoint that searches for Hugging Face datasets matching a query string.
    
    This endpoint allows optional authentication with a Hugging Face token and supports
    different levels of configuration detail in the response. It can fetch basic dataset
    information or include detailed configuration metadata using parallel processing
    for better performance.
    
    HTTP Method: GET
    Route: /api/search/huggingface
    
    Request Headers:
        X-HF-Token (optional): Hugging Face API token for accessing private datasets
        
    Query Parameters:
        query (str, required): Search term to find matching datasets
        limit (int, optional): Maximum number of datasets to return. Defaults to 5.
        include_configs (bool, optional): Whether to include configuration information. 
            Defaults to false.
        config_detail (str, optional): Level of configuration detail to include. 
            Options: 'none', 'basic', 'full'. Defaults to 'basic'.
            
    Returns:
        JSON response containing either:
        - A list of formatted Hugging Face datasets matching the search query
        - When include_configs=true, datasets include configuration information based on
          the config_detail level
        - An error message with appropriate HTTP status code (400 for missing query,
          500 for other errors)
        
    Note:
        - For 'full' config_detail, only processes up to 5 configurations per dataset
        - Uses concurrent processing to improve performance when fetching configuration details
        - Results are cached for 5 minutes to improve performance
    """
    token = request.headers.get('X-HF-Token')
    query = request.args.get('query')
    include_configs = request.args.get(
        'include_configs', 'false').lower() == 'true'
    config_detail_level = request.args.get(
        'config_detail', 'basic')  # Options: none, basic, full

    if not query:
        return jsonify({'error': 'Query parameter is required'}), 400

    if token:
        os.environ['HF_TOKEN'] = token

    limit = int(request.args.get('limit', 5))
    result = search_huggingface_datasets(query, limit)

    if isinstance(result, dict) and 'error' in result:
        return jsonify(result), 500

    if not include_configs or config_detail_level == 'none':
        return jsonify(result)

    def process_dataset_configs(dataset):
        """
    Processes configuration information for a Hugging Face dataset.
    
    This helper function retrieves configuration names for a dataset and, based on the
    specified detail level, fetches additional metadata for each configuration. It handles
    error cases and limits the number of configurations processed for performance reasons.
    
    Args:
        dataset (dict): A dictionary containing dataset information, including the 'id' key
            
    Returns:
        dict: The enhanced dataset dictionary with added configuration information:
            - configs: List of configuration objects with metadata based on detail level
            - total_configs: Total number of configs if there are more than displayed
            - additional_configs: Number of configurations not processed when using full detail
            - configs_error: Error message if configuration processing failed
            
    Note:
        - For 'basic' detail level, only configuration names are included
        - For 'full' detail level, uses parallel processing to fetch detailed information
          but limits to processing only the first 5 configurations
        - This function is primarily used internally by the search_huggingface_endpoint
    """
        try:
            dataset_id = dataset.get('id')
            if not dataset_id:
                return dataset

            configs = get_dataset_config_names(dataset_id)
            if not configs:
                dataset['configs'] = []
                return dataset

            if config_detail_level == 'basic':
                dataset['configs'] = [{'name': config} for config in configs]
                if len(configs) > 5:
                    dataset['total_configs'] = len(configs)
                return dataset

            max_configs_to_process = 5
            configs_to_process = configs[:max_configs_to_process]

            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                future_to_config = {
                    executor.submit(fetch_config_info, dataset_id, config): config
                    for config in configs_to_process
                }

                configs_info = []
                for future in concurrent.futures.as_completed(future_to_config):
                    configs_info.append(future.result())

            if len(configs) > max_configs_to_process:
                dataset['additional_configs'] = len(
                    configs) - max_configs_to_process

            dataset['configs'] = configs_info
            return dataset

        except Exception as e:
            dataset['configs_error'] = str(e)
            return dataset

    with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(result), 3)) as executor:
        processed_datasets = list(executor.map(
            process_dataset_configs, result))

    return jsonify(processed_datasets)


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

    if source.lower() in ['all', 'huggingface']:
        if token:
            os.environ['HF_TOKEN'] = token
        result['huggingface'] = search_huggingface_datasets(query, limit)

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

        elif source.lower() == 'huggingface':
            response = requests.get(
                'https://huggingface.co/api/datasets',
                params={'search': query, 'limit': limit}
            )
            if response.status_code == 200:
                datasets = response.json()
                suggestions = [dataset.get('id') for dataset in datasets]

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

    if source.lower() in ['all', 'kaggle']:
        username = request.headers.get('X-Kaggle-Username')
        key = request.headers.get('X-Kaggle-Key')
        if username and key:
            authenticate_kaggle(username, key)
            result['kaggle'] = get_dataset_suggestions(query, 'kaggle', limit)
        else:
            result['kaggle'] = {'error': 'Kaggle credentials required'}

    if source.lower() in ['all', 'huggingface']:
        token = request.headers.get('X-HF-Token')
        if token:
            os.environ['HF_TOKEN'] = token
        result['huggingface'] = get_dataset_suggestions(
            query, 'huggingface', limit)

    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True, port=5000)
