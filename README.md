# Deep Sage Backend

Deep Sage Backend is a Flask-based API that serves as an intermediary for accessing, downloading, and analyzing data from Kaggle.

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- Kaggle account with API key
- AWS account with S3 bucket (for S3 operations)
- MongoDB (for dataset metadata storage)

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/Deep-Sage-Backend.git
cd Deep-Sage-Backend
```

2. Create a virtual environment:

```bash
# On Windows
python -m venv venv
venv\Scripts\activate

# On macOS/Linux
python -m venv venv
source venv/bin/activate
```

3. Install required packages:

```bash
pip install -r requirements.txt
```

## Kaggle API Setup

The backend requires Kaggle API credentials to access datasets:

1. Create a Kaggle account at [kaggle.com](https://www.kaggle.com/) if you don't have one
2. Go to your account settings: https://www.kaggle.com/settings
3. In the "API" section, click "Create New API Token"
4. This will download a `kaggle.json` file containing your credentials

### Placing kaggle.json (OS-specific instructions)

#### Windows:
```bash
mkdir %USERPROFILE%\.kaggle
copy path\to\downloaded\kaggle.json %USERPROFILE%\.kaggle\
```

#### macOS:
```bash
mkdir -p ~/.kaggle
cp path/to/downloaded/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

#### Linux:
```bash
mkdir -p ~/.kaggle
cp path/to/downloaded/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

## Environment Variables Setup

Create a .env file in the root directory of the project:

```
KAGGLE_USERNAME=your_kaggle_username
KAGGLE_KEY=your_kaggle_api_key
OPEN_AI_API_KEY=your_openai_api_key  # Optional: only needed if using OpenAI features
AWS_ACCESS_KEY=your_aws_access_key   # Required for AWS S3 operations
AWS_SECRET_KEY=your_aws_secret_key   # Required for AWS S3 operations
AWS_BUCKET_KEY=your_bucket_name      # Name of your S3 bucket
REGION=your_aws_region               # e.g., us-east-1, ap-south-1
MONGO_URI=your_mongodb_uri           # MongoDB connection URI (e.g., mongodb://localhost:27017)
```

Replace the placeholder values with your actual credentials:
- `your_kaggle_username`: Your Kaggle account username
- `your_kaggle_api_key`: Your Kaggle API key from the kaggle.json file
- `your_openai_api_key`: Your OpenAI API key (if using OpenAI features)
- `your_aws_access_key`: Your AWS IAM Access Key
- `your_aws_secret_key`: Your AWS IAM Secret Key
- `your_bucket_name`: The name of your S3 bucket
- `your_aws_region`: The AWS region where your S3 bucket is located
- `your_mongodb_uri`: Your MongoDB connection URI

## MongoDB Setup

The application uses MongoDB to store dataset metadata. Make sure you have MongoDB installed and running:

1. Install MongoDB from [mongodb.com](https://www.mongodb.com/try/download/community) if you haven't already
2. Start the MongoDB service
3. Set the `MONGO_URI` environment variable in your .env file (default: `mongodb://localhost:27017`)

## Running the Application

Start the Flask server:

```bash
python app.py
```

The server will start on `http://localhost:5000` by default.

## API Documentation

Once the server is running, you can access the API documentation at:
```
http://localhost:5000/api/docs
```

## Main Features

- Search and browse Kaggle datasets with filtering options
- Download datasets with real-time progress tracking
- Cancel in-progress downloads
- Convert data between formats (Excel to CSV)
- Generate data visualizations
- Get AI-powered insights on datasets
- Upload and manage datasets in AWS S3
- Upload and retrieve user profile photos
- Store and retrieve dataset metadata in MongoDB

## AWS S3 Operations

The backend provides several endpoints for working with AWS S3:

- Upload datasets to S3 (`/api/aws/s3/upload-dataset`)
- List user datasets in S3 (`/api/aws/s3/list-datasets`)
- List S3 buckets (`/api/aws/s3/list-buckets`)
- Download datasets from S3 (`/api/aws/s3/download-dataset`)
- Delete datasets from S3 (`/api/aws/s3/delete-dataset`)
- Record dataset metadata in MongoDB (`/api/aws/s3/record-dataset`)
- Retrieve recorded datasets (`/api/aws/s3/get-recorded-datasets`)
- Download a recorded dataset (`/api/aws/s3/download-recorded-dataset`)

## User Profile Management

- Upload user profile photos to S3 (`/api/upload-profile-photo`)
- Retrieve user profile photo URLs (`/api/aws/s3/get-profile-photo/<user_id>`)

## API Health Check

To verify that the API is running correctly:
```
GET http://localhost:5000/health
```

## Troubleshooting

### Common Issues:

1. **Authentication Errors**:
   - Verify your Kaggle credentials in the .env file
   - Ensure kaggle.json is in the correct location
   - Check that kaggle.json has the proper permissions (600)
   - Verify your AWS credentials in the .env file

2. **Module Import Errors**:
   - Make sure all dependencies are installed: `pip install -r requirements.txt`
   - Verify you're in the virtual environment

3. **Permission Errors**:
   - On Linux/macOS, ensure the download directories have write permissions
   - Verify your AWS IAM user has the correct permissions for S3 operations

4. **Timeout Issues**:
   - Large dataset downloads may timeout; adjust your firewall settings
   - Check your internet connection

5. **MongoDB Connection Issues**:
   - Ensure MongoDB is running
   - Check that the connection string in the .env file is correct

## Security Note

- The .env file contains sensitive credentials - never commit it to version control
- The .gitignore file is configured to exclude this file by default
- Rotate your AWS access keys periodically for enhanced security
