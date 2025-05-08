import boto3
from flask import Blueprint, request, jsonify
import os
from pymongo import MongoClient
from datetime import datetime
import logging

user_sync_bp = Blueprint('user_sync', __name__)

mongo_uri = os.environ.get('MONGO_URI', 'mongodb://localhost:27017')
client = MongoClient(mongo_uri)
db = client.deep_sage_db
dataset_collection = db.datasets


@user_sync_bp.route('/api/aws/s3/record-dataset', methods=['POST'])
def record_dataset():
    """
    Record dataset metadata in MongoDB

    Expected JSON payload:
    {
        "user_id": "user123",
        "dataset_name": "sales_data_2024",
        "s3_path": "s3://bucket-name/path/to/file.csv",
        "file_size": 1024000,
        "file_type": "csv"
    }
    """
    try:
        data = request.json

        required_fields = ['user_id', 'dataset_name',
                           's3_path', 'file_size', 'file_type']
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing required field: {field}"}), 400

        dataset_record = {
            "user_id": data['user_id'],
            "dataset_name": data['dataset_name'],
            "s3_path": data['s3_path'],
            "file_size": data['file_size'],
            "file_type": data['file_type'],
            "last_modified": datetime.now()
        }

        result = dataset_collection.insert_one(dataset_record)

        return jsonify({
            "message": "Dataset recorded successfully",
            "dataset_id": str(result.inserted_id)
        }), 201

    except Exception as e:
        logging.error(f"Error recording dataset: {str(e)}")
        return jsonify({"error": "Failed to record dataset"}), 500


@user_sync_bp.route('/api/aws/s3/get-recorded-datasets', methods=['GET'])
def get_datasets():
    """
    Retrieve datasets for a specific user

    Query parameters:
    - user_id: ID of the user whose datasets to retrieve
    """
    try:
        user_id = request.args.get('user_id')

        if not user_id:
            return jsonify({"error": "Missing required query parameter: user_id"}), 400

        datasets = list(dataset_collection.find({"user_id": user_id},
                                                {"_id": 0}))

        for dataset in datasets:
            if "last_modified" in dataset:
                dataset["last_modified"] = dataset["last_modified"].isoformat()

        return jsonify({
            "user_id": user_id,
            "count": len(datasets),
            "datasets": datasets
        }), 200

    except Exception as e:
        logging.error(f"Error retrieving datasets: {str(e)}")
        return jsonify({"error": "Failed to retrieve datasets"}), 500

@user_sync_bp.route('/api/aws/s3/download-recorded-dataset', methods=['POST'])
def download_recorded_dataset():
    """
    Download a dataset that is recorded in MongoDB
    
    Expected JSON payload:
    {
        "user_id": "user123",
        "s3_path": "s3://bucket-name/datasets/user123/file.csv",
        "destination_path": "/path/to/save/file.csv"
    }
    """
    try:
        data = request.json
        
        required_fields = ['user_id', 's3_path', 'destination_path']
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing required field: {field}"}), 400
        
        user_id = data['user_id']
        s3_path = data['s3_path']
        destination_path = data['destination_path']
        
        # Verify the dataset exists in MongoDB for this user
        dataset = dataset_collection.find_one({
            "user_id": user_id,
            "s3_path": s3_path
        })
        
        if not dataset:
            return jsonify({"error": "Dataset not found for this user"}), 404
        
        # Parse S3 path: s3://bucket-name/path/to/file
        if not s3_path.startswith('s3://'):
            return jsonify({"error": "Invalid S3 path format"}), 400
            
        s3_parts = s3_path[5:].split('/', 1)
        if len(s3_parts) != 2:
            return jsonify({"error": "Invalid S3 path format"}), 400
            
        bucket_name = s3_parts[0]
        object_key = s3_parts[1]
        
        aws_access_key = os.environ.get('AWS_ACCESS_KEY')
        aws_secret_key = os.environ.get('AWS_SECRET_KEY')
        aws_region = os.environ.get('REGION', 'ap-south-1')
        
        if not aws_access_key or not aws_secret_key:
            return jsonify({"error": "AWS credentials not configured on server"}), 500
        
        destination_dir = os.path.dirname(destination_path)
        os.makedirs(destination_dir, exist_ok=True)
        
        s3_client = boto3.client(
            's3',
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_key,
            region_name=aws_region
        )
        
        s3_client.download_file(
            Bucket=bucket_name,
            Key=object_key,
            Filename=destination_path
        )
        
        file_size = os.path.getsize(destination_path)
        
        return jsonify({
            "message": "Dataset downloaded successfully",
            "user_id": user_id,
            "dataset_name": dataset.get("dataset_name"),
            "destination_path": destination_path,
            "file_size": file_size
        }), 200
        
    except boto3.exceptions.ClientError as e:
        error_code = e.response.get('Error', {}).get('Code', 'Unknown')
        if error_code == 'NoSuchKey':
            return jsonify({"error": "File not found in S3"}), 404
        elif error_code in ['InvalidAccessKeyId', 'SignatureDoesNotMatch']:
            return jsonify({"error": "Invalid AWS credentials"}), 401
        return jsonify({"error": str(e)}), 500
    except Exception as e:
        logging.error(f"Error downloading dataset: {str(e)}")
        return jsonify({"error": f"Failed to download dataset: {str(e)}"}), 500