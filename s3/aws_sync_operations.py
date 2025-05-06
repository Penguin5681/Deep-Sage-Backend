import boto3
from flask import Blueprint, request, jsonify
import os
import requests

aws_operations_bp = Blueprint('aws_operations', __name__)


@aws_operations_bp.route('/api/aws/s3/upload-dataset', methods=['PUT'])
def upload_dataset():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file part in the request"}), 400

        file = request.files['file']

        if not file or file.filename == '':
            return jsonify({"error": "No file selected"}), 400

        if 'user_id' not in request.form:
            return jsonify({"error": "User ID is required"}), 400

        user_id = request.form['user_id']

        access_key = request.headers.get('X-AWS-Access-Key')
        secret_key = request.headers.get('X-AWS-Secret-Key')
        region = request.headers.get('X-AWS-Region', 'ap-south-1')
        bucket_name = request.headers.get('X-AWS-Bucket-Name')

        if not access_key or not secret_key or not bucket_name:
            return jsonify({"error": "Missing required headers: X-AWS-Access-Key, X-AWS-Secret-Key, or X-AWS-Bucket-Name"}), 400

        s3_client = boto3.client(
            's3',
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name=region
        )

        s3_path = f"datasets/{user_id}/{file.filename}"

        s3_client.upload_fileobj(file, bucket_name, s3_path, ExtraArgs={
            "ACL": "public-read"
        })

        file_url = f"https://{bucket_name}.s3.{region}.amazonaws.com/{s3_path}"

        dataset_record = {
            'user_id': user_id,
            'dataset_name': file.filename,
            's3_path': f"s3://{bucket_name}/datasets/{user_id}/{file.filename}",
            'file_size': file.content_length,
            'file_type': file.filename.split('.')[1]
        }

        try:
            record_response = requests.post(
                f"{request.host_url.rstrip('/')}/api/aws/s3/record-dataset",
                json=dataset_record,
            )
            record_response.raise_for_status()
        except requests.exceptions.RequestException as req_err:
            print(f"Warning: Failed to record dataset: {req_err}")

        return jsonify({
            "message": "File uploaded successfully",
            "user_id": user_id,
            "file_name": file.filename,
            "s3_path": s3_path,
            "file_url": file_url
        }), 200

    except boto3.exceptions.ClientError as e:
        error_code = e.response.get('Error', {}).get('Code', 'Unknown')
        if error_code == 'InvalidAccessKeyId' or error_code == 'SignatureDoesNotMatch':
            return jsonify({"error": "Invalid AWS credentials"}), 401
        return jsonify({"error": str(e)}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@aws_operations_bp.route('/api/aws/s3/list-datasets', methods=['POST'])
def list_datasets():
    try:
        access_key = request.headers.get('X-AWS-Access-Key')
        secret_key = request.headers.get('X-AWS-Secret-Key')
        region = request.headers.get('X-AWS-Region', 'ap-south-1')
        bucket_name = request.headers.get('X-AWS-Bucket-Name')

        data = request.get_json()
        user_id = data.get('user_id')

        if not access_key or not secret_key or not bucket_name:
            return jsonify({"error": "Missing required headers: X-AWS-Access-Key, X-AWS-Secret-Key, or X-AWS-Bucket-Name"}), 400

        if not user_id:
            return jsonify({"error": "User ID is required"}), 400

        s3_client = boto3.client(
            's3',
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name=region,
        )

        prefix = f"datasets/{user_id}/"

        response = s3_client.list_objects_v2(
            Bucket=bucket_name,
            Prefix=prefix,
            Delimiter='/'
        )

        datasets = []

        if 'Contents' in response:
            for obj in response['Contents']:
                file_name = obj['Key'].split('/')[-1]
                if file_name:
                    file_url = f"https://{bucket_name}.s3.{region}.amazonaws.com/{obj['Key']}"
                    datasets.append({
                        'file_name': file_name,
                        's3_path': obj['Key'],
                        'size': obj['Size'],
                        'last_modified': obj['LastModified'].isoformat(),
                        'file_url': file_url
                    })

        return jsonify({
            "datasets": datasets,
            "user_id": user_id,
            "bucket": bucket_name
        }), 200

    except boto3.exceptions.ClientError as e:
        error_code = e.response.get('Error', {}).get('Code', 'Unknown')
        if error_code == 'InvalidAccessKeyId' or error_code == 'SignatureDoesNotMatch':
            return jsonify({"error": "Invalid AWS credentials"}), 401
        return jsonify({"error": str(e)}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@aws_operations_bp.route('/api/aws/s3/list-buckets', methods=['POST'])
def list_buckets():
    try:
        access_key = request.headers.get('X-AWS-Access-Key')
        secret_key = request.headers.get('X-AWS-Secret-Key')
        region = request.headers.get('X-AWS-Region', 'ap-south-1')

        if not access_key or not secret_key:
            return jsonify({"error": "Missing required headers: X-AWS-Access-Key and/or X-AWS-Secret-Key"}), 400

        s3_client = boto3.client(
            's3',
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name=region,
        )

        response = s3_client.list_buckets()

        filtered_bucket_names = []
        for bucket in response['Buckets']:
            bucket_location = s3_client.get_bucket_location(
                Bucket=bucket['Name'])
            bucket_region = bucket_location.get(
                'LocationConstraint') or 'ap-south-1'

            if bucket_region == region:
                filtered_bucket_names.append(bucket['Name'])

        return jsonify({
            "buckets": filtered_bucket_names,
            "region": region
        })

    except boto3.exceptions.ClientError as e:
        error_code = e.response.get('Error', {}).get('Code', 'Unknown')
        if error_code == 'InvalidAccessKeyId' or error_code == 'SignatureDoesNotMatch':
            return jsonify({"error": "Invalid AWS credentials"}), 401
        return jsonify({"error": str(e)}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@aws_operations_bp.route('/api/aws/s3/download-dataset', methods=['POST'])
def download_dataset():
    try:
        access_key = request.headers.get('X-AWS-Access-Key')
        secret_key = request.headers.get('X-AWS-Secret-Key')
        region = request.headers.get('X-AWS-Region', 'ap-south-1')
        bucket_name = request.headers.get('X-AWS-Bucket-Name')

        data = request.get_json()
        s3_path = data.get('s3_path')
        local_path = data.get('local_path')

        if not access_key or not secret_key or not bucket_name:
            return jsonify({"error": "Missing required headers: X-AWS-Access-Key, X-AWS-Secret-Key, or X-AWS-Bucket-Name"}), 400

        if not s3_path or not local_path:
            return jsonify({"error": "s3_path and local_path are required"}), 400

        s3_client = boto3.client(
            's3',
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name=region,
        )

        # Ensure the directory exists
        os.makedirs(os.path.dirname(local_path), exist_ok=True)

        s3_client.download_file(bucket_name, s3_path, local_path)

        return jsonify({
            "message": "File downloaded successfully",
            "s3_path": s3_path,
            "local_path": local_path
        }), 200

    except boto3.exceptions.ClientError as e:
        error_code = e.response.get('Error', {}).get('Code', 'Unknown')
        if error_code == 'InvalidAccessKeyId' or error_code == 'SignatureDoesNotMatch':
            return jsonify({"error": "Invalid AWS credentials"}), 401
        elif error_code == 'NoSuchKey':
            return jsonify({"error": "File not found in S3"}), 404
        return jsonify({"error": str(e)}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@aws_operations_bp.route('/api/aws/s3/delete-dataset', methods=['DELETE'])
def delete_dataset():
    try:
        access_key = request.headers.get('X-AWS-Access-Key')
        secret_key = request.headers.get('X-AWS-Secret-Key')
        region = request.headers.get('X-AWS-Region', 'ap-south-1')
        bucket_name = request.headers.get('X-AWS-Bucket-Name')

        data = request.get_json()
        s3_path = data.get('s3_path')

        if not access_key or not secret_key or not bucket_name:
            return jsonify({"error": "Missing required headers: X-AWS-Access-Key, X-AWS-Secret-Key, or X-AWS-Bucket-Name"}), 400

        if not s3_path:
            return jsonify({"error": "s3_path is required"}), 400

        s3_client = boto3.client(
            's3',
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name=region,
        )

        s3_client.delete_object(
            Bucket=bucket_name,
            Key=s3_path
        )

        return jsonify({
            "message": "File deleted successfully",
            "s3_path": s3_path,
            "bucket": bucket_name
        }), 200

    except boto3.exceptions.ClientError as e:
        error_code = e.response.get('Error', {}).get('Code', 'Unknown')
        if error_code == 'InvalidAccessKeyId' or error_code == 'SignatureDoesNotMatch':
            return jsonify({"error": "Invalid AWS credentials"}), 401
        return jsonify({"error": str(e)}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500
