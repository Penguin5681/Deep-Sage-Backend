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
