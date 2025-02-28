from flask import Flask, jsonify, request
import os
import requests
from kaggle.api.kaggle_api_extended import KaggleApi
from huggingface_hub import HfApi

app = Flask(__name__)

kaggle_api = KaggleApi()
hf_api = HfApi()

def authenticate_kaggle(username, key):
    os.environ['KAGGLE_USERNAME'] = username
    os.environ['KAGGLE_KEY'] = key
    kaggle_api.authenticate()

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

def get_huggingface_datasets(limit=5, sort_by='downloads'):
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
    token = request.headers.get('X-HF-Token')
    if token:
        os.environ['HF_TOKEN'] = token
    
    limit = int(request.args.get('limit', 5))
    sort_by = request.args.get('sort_by', 'downloads')
    result = get_huggingface_datasets(limit, sort_by)
    return jsonify(result)

@app.route('/api/datasets', methods=['GET'])
def get_datasets():
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

@app.route('/api/datasets/download', methods=['POST'])
def download_dataset():
    data = request.json
    source = data.get('source')
    dataset_id = data.get('dataset_id')
    path = data.get('path', './datasets')
    
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
            kaggle_api.dataset_download_files(dataset_id, path, unzip=True)
            return jsonify({'success': True, 'message': f'Dataset {dataset_id} downloaded successfully to {path}'})
        
        elif source.lower() == 'huggingface':
            if token:
                os.environ['HF_TOKEN'] = token
                
            from datasets import load_dataset
            dataset = load_dataset(dataset_id, cache_dir=path)

            if isinstance(dataset, dict):
                for split_name, split_dataset in dataset.items():
                    split_dataset.to_pandas().to_csv(f"{path}/{dataset_id.replace('/', '_')}_{split_name}.csv", index=False)
            else:
                dataset.to_pandas().to_csv(f"{path}/{dataset_id.replace('/', '_')}.csv", index=False)
            return jsonify({'success': True, 'message': f'Dataset {dataset_id} downloaded successfully to {path}'})
        
        else:
            return jsonify({'error': 'Invalid source. Use "kaggle" or "huggingface"'}), 400
        
    except Exception as e:
        return jsonify({'error': f'Error downloading dataset: {str(e)}'}), 500
    
def search_kaggle_datasets(query, limit=5):
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

def search_huggingface_datasets(query, limit=5):
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

@app.route('/api/search/huggingface', methods=['GET'])
def search_huggingface_endpoint():
    token = request.headers.get('X-HF-Token')
    query = request.args.get('query')
    
    if not query:
        return jsonify({'error': 'Query parameter is required'}), 400
    
    if token:
        os.environ['HF_TOKEN'] = token
    
    limit = int(request.args.get('limit', 5))
    result = search_huggingface_datasets(query, limit)
    return jsonify(result)

@app.route('/api/search', methods=['GET'])
def search_all_datasets():
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

def get_dataset_suggestions(query, source, limit=5):
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
        result['huggingface'] = get_dataset_suggestions(query, 'huggingface', limit)
    
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, port=5000)