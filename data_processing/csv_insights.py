import os
import io
import pandas as pd
import numpy as np
from flask import Blueprint, request, jsonify
from werkzeug.utils import secure_filename
import json
from dotenv import load_dotenv
import openai
from typing import Dict, List, Any, Optional

# Load environment variables
load_dotenv()


csv_insights_api = Blueprint('csv_insights', __name__)

def analyze_csv(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze CSV data and provide basic insights using pandas
    """
    total_rows, total_columns = df.shape
    insights = {
        "row_count": total_rows,
        "column_count": total_columns,
        "columns": list(df.columns),
        "missing_values": {},
        "duplicates": {"count": 0, "percentage": 0},
        "data_types": {},
        "numeric_stats": {},
        "categorical_stats": {},
        "outliers": {},
        "data_quality": {}  # New section for quality metrics
    }

    # Check missing values
    missing_values = df.isnull().sum()
    missing_percentage = (missing_values / total_rows) * 100

    # Overall missing values metrics
    insights["data_quality"]["total_missing_cells"] = int(missing_values.sum())
    insights["data_quality"]["total_missing_percentage"] = round((missing_values.sum() / (total_rows * total_columns)) * 100, 2)

    if missing_values.sum() > 0:
        for column in df.columns:
            missing_count = missing_values[column]
            if missing_count > 0:
                insights["missing_values"][column] = {
                    "count": int(missing_count),
                    "percentage": round(missing_percentage[column], 2)
                }

    # Check duplicates
    duplicate_count = df.duplicated().sum()
    insights["duplicates"]["count"] = int(duplicate_count)
    insights["duplicates"]["percentage"] = round((duplicate_count / total_rows) * 100, 2) if total_rows > 0 else 0

    # Analyze data types
    type_counts = {"numeric": 0, "categorical": 0, "datetime": 0, "other": 0}

    for column in df.columns:
        insights["data_types"][column] = str(df[column].dtype)

        # Numeric stats for numeric columns
        if pd.api.types.is_numeric_dtype(df[column]):
            type_counts["numeric"] += 1
            insights["numeric_stats"][column] = {
                "min": float(df[column].min()) if not pd.isna(df[column].min()) else None,
                "max": float(df[column].max()) if not pd.isna(df[column].max()) else None,
                "mean": float(df[column].mean()) if not pd.isna(df[column].mean()) else None,
                "median": float(df[column].median()) if not pd.isna(df[column].median()) else None,
                "std": float(df[column].std()) if not pd.isna(df[column].std()) else None
            }

            # Detect outliers using IQR
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            outlier_count = ((df[column] < (Q1 - 1.5 * IQR)) | (df[column] > (Q3 + 1.5 * IQR))).sum()
            if outlier_count > 0:
                insights["outliers"][column] = {
                    "count": int(outlier_count),
                    "percentage": round((outlier_count / total_rows) * 100, 2)
                }

        # Categorical stats
        elif pd.api.types.is_object_dtype(df[column]) or pd.api.types.is_categorical_dtype(df[column]):
            type_counts["categorical"] += 1
            value_counts = df[column].value_counts()
            unique_count = len(value_counts)
            insights["categorical_stats"][column] = {
                "unique_values": int(unique_count),
                "top_5_values": value_counts.head(5).to_dict(),
            }

        # Datetime detection
        elif pd.api.types.is_datetime64_any_dtype(df[column]):
            type_counts["datetime"] += 1
        else:
            type_counts["other"] += 1

    # Calculate quality scores
    # Data completeness (0-100): 100 means no missing values
    insights["data_quality"]["completeness_score"] = round(100 - insights["data_quality"]["total_missing_percentage"], 2)

    # Duplication quality (0-100): 100 means no duplicates
    insights["data_quality"]["duplication_score"] = round(100 - insights["duplicates"]["percentage"], 2)

    # Outlier score (0-100): 100 means no outliers
    total_numeric_outliers = sum(outlier["count"] for outlier in insights["outliers"].values()) if insights["outliers"] else 0
    total_numeric_cells = total_rows * type_counts["numeric"] if type_counts["numeric"] > 0 else 1
    outlier_percentage = (total_numeric_outliers / total_numeric_cells) * 100 if total_numeric_cells > 0 else 0
    insights["data_quality"]["outlier_score"] = round(100 - outlier_percentage, 2)

    # Overall quality score (average of above metrics)
    insights["data_quality"]["overall_score"] = round(
        (insights["data_quality"]["completeness_score"] + 
         insights["data_quality"]["duplication_score"] + 
         insights["data_quality"]["outlier_score"]) / 3, 2
    )

    # Add column-level quality scores
    insights["data_quality"]["column_scores"] = {}
    for column in df.columns:
        # Calculate column quality score based on missing values and outliers
        missing_pct = missing_percentage[column] if column in missing_percentage else 0
        completeness = 100 - missing_pct

        outlier_pct = 0
        if pd.api.types.is_numeric_dtype(df[column]) and column in insights["outliers"]:
            outlier_pct = insights["outliers"][column]["percentage"]

        outlier_quality = 100 - outlier_pct
        column_score = round((completeness + outlier_quality) / 2, 2)

        insights["data_quality"]["column_scores"][column] = {
            "completeness": round(completeness, 2),
            "outlier_quality": round(outlier_quality, 2),
            "overall": column_score
        }

    # Add column type distribution
    insights["data_quality"]["column_types"] = type_counts

    return insights

def get_ai_insights(df: pd.DataFrame, basic_insights: Dict[str, Any]) -> str:
    """
    Get advanced insights using OpenAI's GPT-4o mini
    """
    # Prepare a summary of the data for the AI
    column_info = []
    for col in df.columns:
        if col in basic_insights["numeric_stats"]:
            col_type = "numeric"
            stats = basic_insights["numeric_stats"][col]
        elif col in basic_insights["categorical_stats"]:
            col_type = "categorical"
            stats = basic_insights["categorical_stats"][col]
        else:
            col_type = basic_insights["data_types"][col]
            stats = {}

        missing = basic_insights["missing_values"].get(col, {"count": 0, "percentage": 0})
        column_info.append(f"- {col} ({col_type}): {missing['count']} missing values ({missing['percentage']}%)")

    data_summary = "\n".join(column_info)

    # Create a prompt for the AI
    prompt = f"""
    Analyze this dataset summary and provide actionable insights and recommendations:
    
    Dataset: {basic_insights['row_count']} rows, {basic_insights['column_count']} columns
    Duplicates: {basic_insights['duplicates']['count']} rows ({basic_insights['duplicates']['percentage']}%)
    
    Column information:
    {data_summary}
    
    Provide insights focused on:
    1. Data quality issues and how to address them
    2. Recommendations for handling missing values based on the column type
    3. Potential features or transformations that might be valuable
    4. Any other helpful observations about the dataset structure
    
    Keep your response concise and actionable.
    """

    try:
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a data analysis assistant specializing in CSV file insights."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error getting AI insights: {str(e)}"

@csv_insights_api.route('/api/analyze-csv', methods=['POST'])
def analyze_csv_file():
    # Check if file is present in the request
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']

    # Check if file is selected
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    # Check file extension
    if not file.filename.endswith('.csv'):
        return jsonify({'error': 'Only CSV files are supported'}), 400

    # Get the use_ai parameter
    use_ai = request.args.get('use_ai', 'false').lower() == 'true'

    try:
        # Read the uploaded CSV file
        content = file.read()
        df = pd.read_csv(io.BytesIO(content))

        # Get basic insights
        basic_insights = analyze_csv(df)

        response = {"basic_insights": basic_insights}

        # Get AI insights if requested
        if use_ai:
            # For large files, we might want to run AI analysis asynchronously
            # but Flask doesn't have built-in async support like FastAPI
            # For now, we'll process synchronously
            if len(df) > 10000:
                response["advanced_insights"] = "AI insights are not available for large files in this version."
            else:
                ai_insights = get_ai_insights(df, basic_insights)
                response["advanced_insights"] = ai_insights

        return jsonify(response)

    except Exception as e:
        return jsonify({'error': f"Error analyzing CSV: {str(e)}"}), 500