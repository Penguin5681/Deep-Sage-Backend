# Deep Sage Backend

A Flask-based API service for accessing and downloading datasets from Kaggle and Hugging Face.

## Overview

Deep Sage Backend provides a unified interface to search and download datasets from popular machine learning repositories. It simplifies the process of discovering and acquiring datasets for your data science and machine learning projects.

## Requirements

- Python 3.7+
- Flask
- Kaggle API
- Hugging Face Hub
- requests

## Installation

```bash
# Clone the repository
git clone https://github.com/Penguin5681/Deep-Sage-Backend.git
cd Deep-Sage-Backend

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Configuration

To use this API, you'll need:

1. **Kaggle API credentials** - Get them from your Kaggle account settings
2. **Hugging Face token** (optional) - For accessing private datasets

## API Endpoints

### 1. Get Datasets from Multiple Sources

```
GET /api/datasets
```

Retrieve datasets from Kaggle, Hugging Face, or both.

**Query Parameters:**
- `source` (optional): Data source to query (`all`, `kaggle`, or `huggingface`). Default: `all`
- `limit` (optional): Maximum number of datasets to return. Default: `5`
- `kaggle_sort` (optional): Sort method for Kaggle results (`hottest`, `votes`, `updated`, `active`, `published`). Default: `hottest`
- `hf_sort` (optional): Sort method for Hugging Face results (`downloads`, `trending`, `modified`). Default: `downloads`

**Headers:**
- `X-Kaggle-Username`: Your Kaggle username (required for Kaggle datasets)
- `X-Kaggle-Key`: Your Kaggle API key (required for Kaggle datasets)
- `X-HF-Token`: Your Hugging Face token (optional)

**Response:**
```json
{
    "kaggle": [
        {
            "id": "owner/dataset-name",
            "title": "Dataset Title",
            "owner": "owner",
            "url": "https://www.kaggle.com/datasets/owner/dataset-name",
            "size": "1.2GB",
            "lastUpdated": "2023-04-01",
            "downloadCount": 5000,
            "voteCount": 250,
            "description": "Dataset description"
        }
    ],
    "huggingface": [
        {
            "id": "username/dataset-name",
            "author": "username",
            "url": "https://huggingface.co/datasets/username/dataset-name",
            "downloads": 10000,
            "likes": 500,
            "lastModified": "2023-05-15",
            "tags": ["tag1", "tag2"],
            "description": "Dataset description"
        }
    ]
}
```

### 2. Get Kaggle Datasets

```
GET /api/datasets/kaggle
```

Retrieve datasets from Kaggle only.

**Query Parameters:**
- `limit` (optional): Maximum number of datasets to return. Default: `5`
- `sort_by` (optional): Sort method (`hottest`, `votes`, `updated`, `active`, `published`). Default: `hottest`

**Headers:**
- `X-Kaggle-Username`: Your Kaggle username (required)
- `X-Kaggle-Key`: Your Kaggle API key (required)

### 3. Get Hugging Face Datasets

```
GET /api/datasets/huggingface
```

Retrieve datasets from Hugging Face only.

**Query Parameters:**
- `limit` (optional): Maximum number of datasets to return. Default: `5`
- `sort_by` (optional): Sort method (`downloads`, `trending`, `modified`). Default: `downloads`

**Headers:**
- `X-HF-Token`: Your Hugging Face token (optional)

### 4. Download Dataset

```
POST /api/datasets/download
```

Download a dataset from the specified source.

**Headers:**
- `X-Kaggle-Username`: Your Kaggle username (required for Kaggle datasets)
- `X-Kaggle-Key`: Your Kaggle API key (required for Kaggle datasets)
- `X-HF-Token`: Your Hugging Face token (optional)

**Request Body:**
```json
{
    "source": "kaggle",  // or "huggingface"
    "dataset_id": "owner/dataset-name",
    "path": "./datasets"  // Optional, default is "./datasets"
}
```

**Response:**
```json
{
    "success": true,
    "message": "Dataset owner/dataset-name downloaded successfully to ./datasets"
}
```

## Error Handling

All endpoints return appropriate HTTP status codes and error messages:

- `400` - Bad Request (e.g., missing required parameters)
- `401` - Unauthorized (e.g., missing or invalid credentials)
- `500` - Server Error (with error message)

## Running the Application

```bash
python app.py
```

The server will start on `http://localhost:5000`.

## License

[Your License Here]