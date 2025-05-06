# Deep Sage Backend

Deep Sage Backend is a Flask-based API that serves as an intermediary for accessing, downloading, and analyzing data from Kaggle.

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- Kaggle account with API key

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
```

Replace the placeholder values with your actual credentials:
- `your_kaggle_username`: Your Kaggle account username
- `your_kaggle_api_key`: Your Kaggle API key from the kaggle.json file
- `your_openai_api_key`: Your OpenAI API key (if using OpenAI features)

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

2. **Module Import Errors**:
   - Make sure all dependencies are installed: `pip install -r requirements.txt`
   - Verify you're in the virtual environment

3. **Permission Errors**:
   - On Linux/macOS, ensure the download directories have write permissions

4. **Timeout Issues**:
   - Large dataset downloads may timeout; adjust your firewall settings
   - Check your internet connection

## Security Note

- The .env file contains sensitive credentials - never commit it to version control
- The .gitignore file is configured to exclude this file by default