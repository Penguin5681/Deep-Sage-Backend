import os
import uuid
import csv
import json
import re
from json import JSONDecodeError
from typing import Dict, List, Tuple, Any
from flask import Blueprint, request, jsonify, current_app, send_from_directory
from werkzeug.utils import secure_filename
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPEN_AI_API_KEY"))


finetune_api = Blueprint('finetune_api', __name__)

UPLOAD_FOLDER = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), 'uploads')
OUTPUT_FOLDER = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), 'outputs')


os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


def save_uploaded_file(file) -> Tuple[str, str]:
    """Save an uploaded file with a unique ID and return the file ID and path."""
    file_id = str(uuid.uuid4())
    filename = f"{file_id}_{secure_filename(file.filename)}"
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)
    return file_id, filepath


def read_csv_file(filepath: str) -> Tuple[List[str], List[Dict[str, Any]]]:
    """Read a CSV file and return column names and rows."""
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        column_names = reader.fieldnames or []
        rows = [row for row in reader]
    return column_names, rows


def generate_autoprompt_jsonl(rows: List[Dict[str, Any]], output_filepath: str) -> None:
    """Generate a JSONL file with automatically created prompts and responses."""
    with open(output_filepath, 'w', encoding='utf-8') as f:
        for row in rows:

            prompt = "Can you give me details about this entry?"

            response_parts = [f"{key}: {value}" for key, value in row.items()]
            response = ", ".join(response_parts)

            entry = {
                "messages": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": response}
                ]
            }

            f.write(json.dumps(entry) + '\n')


def generate_template_jsonl(rows: List[Dict[str, Any]], prompt_template: str,
                            response_template: str, output_filepath: str) -> None:
    """Generate a JSONL file using user-defined templates."""
    with open(output_filepath, 'w', encoding='utf-8') as f:
        for row in rows:
            try:

                prompt = prompt_template.format(**row)
                response = response_template.format(**row)

                entry = {
                    "messages": [
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": response}
                    ]
                }

                f.write(json.dumps(entry) + '\n')
            except KeyError as e:
                current_app.logger.error(f"Template error with row {row}: {e}")
                continue


def find_file_by_id(file_id: str, directory: str) -> str:
    """Find a file by its ID in the specified directory."""
    for filename in os.listdir(directory):
        if filename.startswith(file_id):
            return os.path.join(directory, filename)
    return None


@finetune_api.route('/api/finetune/upload_csv', methods=['POST'])
def upload_csv():
    """Upload a CSV file and return column names and preview."""
    if 'file' not in request.files:
        return jsonify({"status": "error", "message": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"status": "error", "message": "No selected file"}), 400

    if not file.filename.endswith('.csv'):
        return jsonify({"status": "error", "message": "File must be a CSV"}), 400

    file_id, filepath = save_uploaded_file(file)
    column_names, rows = read_csv_file(filepath)

    preview = rows[:5] if len(rows) >= 5 else rows

    return jsonify({
        "status": "success",
        "file_id": file_id,
        "column_names": column_names,
        "preview": preview
    })

@finetune_api.route('/api/finetune/generate_autoprompt', methods=['POST'])
def generate_autoprompt():
    """Generate auto-prompted JSONL file from a previously uploaded CSV."""
    data = request.json
    if not data or 'file_id' not in data:
        return jsonify({"status": "error", "message": "File ID is required"}), 400

    file_id = data['file_id']
    filepath = find_file_by_id(file_id, UPLOAD_FOLDER)

    if not filepath:
        return jsonify({"status": "error", "message": "File not found"}), 404

    _, rows = read_csv_file(filepath)

    output_filename = f"{file_id}_autoprompt.jsonl"
    output_filepath = os.path.join(OUTPUT_FOLDER, output_filename)
    generate_autoprompt_jsonl(rows, output_filepath)

    return jsonify({
        "status": "success",
        "download_url": f"/api/finetune/download/{file_id}_autoprompt"
    })


@finetune_api.route('/api/finetune/generate_from_config', methods=['POST'])
def generate_from_config():
    """Generate a JSONL file using user-defined templates."""
    data = request.json
    required_fields = ['file_id', 'prompt_template', 'response_template']

    if not data or not all(field in data for field in required_fields):
        return jsonify({
            "status": "error",
            "message": f"Missing required parameters: {', '.join(required_fields)}"
        }), 400

    file_id = data['file_id']
    prompt_template = data['prompt_template']
    response_template = data['response_template']

    filepath = find_file_by_id(file_id, UPLOAD_FOLDER)

    if not filepath:
        return jsonify({"status": "error", "message": "File not found"}), 404

    _, rows = read_csv_file(filepath)

    output_filename = f"{file_id}_template.jsonl"
    output_filepath = os.path.join(OUTPUT_FOLDER, output_filename)
    generate_template_jsonl(rows, prompt_template,
                            response_template, output_filepath)

    return jsonify({
        "status": "success",
        "download_url": f"/api/finetune/download/{file_id}_template"
    })


@finetune_api.route('/api/finetune/download/<file_id>', methods=['GET'])
def download_file(file_id):
    """Download a generated JSONL file."""
    for filename in os.listdir(OUTPUT_FOLDER):
        if filename.startswith(file_id):
            return send_from_directory(OUTPUT_FOLDER, filename, as_attachment=True)

    return jsonify({"status": "error", "message": "File not found"}), 404

@finetune_api.route('/api/finetune/suggest_templates', methods=['POST'])
def suggest_templates():
    """Suggest prompt/response templates based on a few CSV rows."""
    data = request.json or {}
    if 'file_id' not in data:
        return jsonify({"status": "error", "message": "file_id is required"}), 400

    file_id = data['file_id']
    sample_size = int(data.get('sample_size', 3))
    filepath = find_file_by_id(file_id, UPLOAD_FOLDER)
    if not filepath:
        return jsonify({"status": "error", "message": "File not found"}), 404

    _, rows = read_csv_file(filepath)
    sample = rows[:sample_size]

    user_prompt = (
        "You are a helpful assistant that generates TEMPLATE STRINGS for prompts and responses using "
        "curly-brace placeholders for column names. Do NOT fill in actual values—leave placeholders intact.\n\n"
        "Sample data rows (JSON array):\n"
        f"{json.dumps(sample, indent=2)}\n\n"
        "Provide 3 distinct prompt templates and 3 corresponding response templates. "
        "Use placeholders in the form {ColumnName}, matching the CSV headers exactly. "
        "Return only a JSON array of objects with keys 'prompt_template' and 'response_template'."
    )

    try:
        resp = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that suggests prompt/response templates."},
                {"role": "user",   "content": user_prompt}
            ],
            temperature=0.7,
            max_tokens=400
        )
        raw = resp.choices[0].message.content.strip()

        cleaned = re.sub(r"```(?:json)?\n?|```", "", raw).strip()

        m = re.search(r"\[.*\]", cleaned, flags=re.DOTALL)
        json_text = m.group(0) if m else cleaned

        templates = json.loads(json_text)

    except (JSONDecodeError, Exception) as e:
        current_app.logger.error(f"Template suggestion failed: {e}\nRaw response:\n{raw}")
        return jsonify({
            "status": "error",
            "message": "Failed to parse model response as JSON",
            "detail": str(e)
        }), 500

    return jsonify({
        "status": "success",
        "templates": templates
    })