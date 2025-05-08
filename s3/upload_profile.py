from dotenv import load_dotenv
from flask import Blueprint, jsonify, request
import boto3
import os

load_dotenv()

upload_profile_bp = Blueprint('upload_profile', __name__)

AWS_ACCESS_KEY = os.environ['AWS_ACCESS_KEY']
AWS_SECRET_KEY = os.environ['AWS_SECRET_KEY']
AWS_BUCKET_KEY = os.environ['AWS_BUCKET_KEY']
AWS_BUCKET_REGION = os.environ['REGION']

s3_client = boto3.client(
    's3',
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY,
    region_name=AWS_BUCKET_REGION
)


@upload_profile_bp.route('/api/upload-profile-photo', methods=['POST'])
def upload_profile_pic():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file part in the request"}), 400

        file = request.files['file']

        if not file or file.filename == '':
            return jsonify({"error": "No file selected"}), 400

        if 'user_id' not in request.form:
            return jsonify({"error": "User ID is required"}), 400

        user_id = request.form['user_id']

        filename = f"profile_photos/{user_id}_profile.jpg"

        s3_client.upload_fileobj(
            file,
            AWS_BUCKET_KEY,
            filename,
            ExtraArgs={
                "ACL": "public-read",
                "ContentType": file.content_type
            }
        )

        file_url = f"https://{AWS_BUCKET_KEY}.s3.{AWS_BUCKET_REGION}.amazonaws.com/{filename}"
        return jsonify({"url": file_url}), 200

    except boto3.exceptions.S3UploadFailedError as e:
        return jsonify({"error": f"S3 upload failed: {str(e)}"}), 500
    except boto3.exceptions.Boto3Error as e:
        return jsonify({"error": f"AWS error: {str(e)}"}), 500
    except Exception as e:
        return jsonify({"error": f"Unexpected error: {str(e)}"}), 500


@upload_profile_bp.route('/api/aws/s3/get-profile-photo/<user_id>', methods=['GET'])
def get_profile_photo(user_id):
    try:
        filename = f"profile_photos/{user_id}_profile.jpg"
        file_url = f"https://{AWS_BUCKET_KEY}.s3.{AWS_BUCKET_REGION}.amazonaws.com/{filename}"

        return jsonify({"url": file_url}), 200

    except Exception as e:
        return jsonify({"error": f"Unexpected error: {str(e)}"}), 500
