# Deep Sage Backend

A Flask-based backend service for exploring and downloading datasets from Kaggle and Hugging Face.

## Features

- Search datasets from Kaggle and Hugging Face
- Get popular/trending datasets
- View dataset configurations and information
- Download datasets with specific configurations
- Intelligent search suggestions
- API response caching

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/Penguin5681/deep_sage_backend.git
    cd deep_sage_backend
    ```

2. Install required packages:
    ```bash
    pip install flask kaggle huggingface_hub datasets flask_caching requests
    ```

3. Set up API credentials:
    - For Kaggle: Get your username and API key from your [Kaggle account settings](https://www.kaggle.com/account)
    - For Hugging Face: Get your token from your [Hugging Face settings](https://huggingface.co/settings/tokens)

## Usage

Start the server:
```bash
python app.py
```

The API will be available at `http://localhost:5000`.

## API Endpoints

### 1. Get Trending Datasets

**Endpoint:** `GET /api/datasets`

**Description:** Retrieve trending datasets from Kaggle and/or Hugging Face.

**Query Parameters:**
- `source`: Data source (`all`, `kaggle`, or `huggingface`)
- `limit`: Maximum number of results (default: 5)
- `kaggle_sort`: Sort method for Kaggle (`hottest`, `votes`, `updated`, `active`, `published`)
- `hf_sort`: Sort method for HuggingFace (`downloads`, `trending`, `modified`)

**Headers:**
- `X-Kaggle-Username`: Kaggle username (required for Kaggle)
- `X-Kaggle-Key`: Kaggle API key (required for Kaggle)
- `X-HF-Token`: Hugging Face token (optional)

**Example Request:**
```bash
curl -X GET "http://localhost:5000/api/datasets?source=all&limit=3" \
  -H "X-Kaggle-Username: your_username" \
  -H "X-Kaggle-Key: your_api_key"
```

**Example Response:**
```json
{
  "kaggle": [
    {
      "id": "allen-institute-for-ai/cord-19",
      "title": "COVID-19 Research Dataset",
      "owner": "Allen Institute for AI",
      "url": "https://www.kaggle.com/datasets/allen-institute-for-ai/cord-19",
      "size": "5.78GB",
      "lastUpdated": "2023-05-15",
      "downloadCount": 12500,
      "voteCount": 475,
      "description": "COVID-19 Open Research Dataset"
    }
  ],
  "huggingface": [
    {
      "id": "databricks/dolly-15k",
      "author": "Databricks",
      "url": "https://huggingface.co/datasets/databricks/dolly-15k",
      "downloads": 87642,
      "likes": 312,
      "lastModified": "2023-04-12",
      "tags": ["instruction-tuning", "llm", "text"],
      "description": "Databricks' Dolly-15k, a publicly available human-generated dataset for instruction tuning"
    }
  ]
}
```

### 2. Get Dataset Configurations

**Endpoint:** `GET /api/datasets/configs`

**Description:** Retrieve available configurations for a Hugging Face dataset.

**Query Parameters:**
- `dataset_id`: Hugging Face dataset ID (required)

**Headers:**
- `X-HF-Token`: Hugging Face token (optional)

**Example Request:**
```bash
curl -X GET "http://localhost:5000/api/datasets/configs?dataset_id=glue"
```

**Example Response:**
```json
{
  "configs": [
    {
      "name": "cola",
      "total_size_bytes": 368934,
      "total_examples": 10657
    },
    {
      "name": "sst2",
      "total_size_bytes": 7439256,
      "total_examples": 70042
    },
    {
      "name": "mnli",
      "total_size_bytes": 30968512,
      "total_examples": 433872
    }
  ]
}
```

### 3. Search Datasets

**Endpoint:** `GET /api/search`

**Description:** Search for datasets across Kaggle and/or Hugging Face.

**Query Parameters:**
- `query`: Search text (required)
- `source`: Data source (`all`, `kaggle`, or `huggingface`)
- `limit`: Maximum number of results (default: 5)

**Headers:**
- `X-Kaggle-Username`: Kaggle username (required for Kaggle)
- `X-Kaggle-Key`: Kaggle API key (required for Kaggle)
- `X-HF-Token`: Hugging Face token (optional)

**Example Request:**
```bash
curl -X GET "http://localhost:5000/api/search?query=COVID-19&source=all&limit=2" \
  -H "X-Kaggle-Username: your_username" \
  -H "X-Kaggle-Key: your_api_key"
```

**Example Response:**
```json
{
  "kaggle": [
    {
      "id": "allen-institute-for-ai/cord-19",
      "title": "COVID-19 Research Dataset",
      "owner": "Allen Institute for AI",
      "url": "https://www.kaggle.com/datasets/allen-institute-for-ai/cord-19",
      "size": "5.78GB",
      "lastUpdated": "2023-05-15",
      "downloadCount": 12500,
      "voteCount": 475,
      "description": "COVID-19 Open Research Dataset"
    }
  ],
  "huggingface": [
    {
      "id": "gsarti/covid-nli",
      "author": "Gabriele Sarti",
      "url": "https://huggingface.co/datasets/gsarti/covid-nli",
      "downloads": 5432,
      "likes": 45,
      "lastModified": "2022-09-22",
      "tags": ["covid", "nli", "text"],
      "description": "Natural Language Inference dataset about COVID-19"
    }
  ]
}
```

### 4. Search Hugging Face with Configuration Details

**Endpoint:** `GET /api/search/huggingface`

**Description:** Advanced search for Hugging Face datasets with configuration details.

**Query Parameters:**
- `query`: Search text (required)
- `include_configs`: Include dataset configurations (`true` or `false`)
- `config_detail`: Level of configuration detail (`none`, `basic`, or `full`)
- `limit`: Maximum number of results (default: 5)

**Headers:**
- `X-HF-Token`: Hugging Face token (optional)

**Example Request:**
```bash
curl -X GET "http://localhost:5000/api/search/huggingface?query=glue&include_configs=true&config_detail=basic&limit=1"
```

**Example Response:**
```json
{
  "datasets": [
    {
      "id": "glue",
      "author": "HuggingFace",
      "url": "https://huggingface.co/datasets/glue",
      "downloads": 152689,
      "likes": 98,
      "lastModified": "2023-03-17",
      "tags": ["benchmark", "nlp", "evaluation"],
      "description": "GLUE benchmark for evaluating natural language understanding systems",
      "configs": [
        {
          "name": "cola",
          "total_size_bytes": 368934,
          "total_examples": 10657
        },
        {
          "name": "sst2",
          "total_size_bytes": 7439256,
          "total_examples": 70042
        }
      ]
    }
  ]
}
```

### 5. Dataset Download

**Endpoint:** `POST /api/datasets/download`

**Description:** Download a dataset from Kaggle or Hugging Face.

**Request Body:**
```json
{
  "source": "huggingface",
  "dataset_id": "dataset_name",
  "path": "./downloads",
  "config": "config_name"
}
```

**Headers:**
- `X-Kaggle-Username`: Kaggle username (required for Kaggle)
- `X-Kaggle-Key`: Kaggle API key (required for Kaggle)
- `X-HF-Token`: Hugging Face token (optional)

**Example Request:**
```bash
curl -X POST "http://localhost:5000/api/datasets/download" \
  -H "Content-Type: application/json" \
  -d '{
    "source": "huggingface",
    "dataset_id": "glue",
    "path": "./downloads",
    "config": "cola"
  }'
```

**Example Response:**
```json
{
  "status": "success",
  "message": "Dataset glue with config cola downloaded successfully",
  "path": "./downloads/glue/cola"
}
```

### 6. Auto-complete Suggestions

**Endpoint:** `GET /api/suggestions`

**Description:** Get auto-complete suggestions for partial search queries.

**Query Parameters:**
- `query`: Partial search text (required)
- `source`: Data source (`all`, `kaggle`, or `huggingface`)
- `limit`: Maximum number of results (default: 5)

**Headers:**
- `X-Kaggle-Username`: Kaggle username (required for Kaggle)
- `X-Kaggle-Key`: Kaggle API key (required for Kaggle)
- `X-HF-Token`: Hugging Face token (optional)

**Example Request:**
```bash
curl -X GET "http://localhost:5000/api/suggestions?query=cov&source=all&limit=3" \
  -H "X-Kaggle-Username: your_username" \
  -H "X-Kaggle-Key: your_api_key"
```

**Example Response:**
```json
{
  "suggestions": [
    "covid-19",
    "covid vaccine",
    "covariance matrix"
  ]
}
```

## Error Handling

The API returns appropriate HTTP status codes with error messages:

```json
{
  "error": "Error message description"
}
```

## Caching

The application implements caching (5 minutes default) for search results and configuration data to improve performance.

## License

MIT