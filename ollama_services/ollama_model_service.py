import threading
import re
from flask import Blueprint, jsonify, request, current_app
import csv
import os
import subprocess
import signal

ollama_model_bp = Blueprint('ollama_model', __name__)

download_status: dict[str, float] = {}
download_procs: dict[str, subprocess.Popen] = {}

@ollama_model_bp.route('/api/ollama/get-available-models', methods=['GET'])
def get_available_models():
    """Return the contents of data.csv as JSON."""
    csv_path = os.path.join(os.path.dirname(__file__), 'data.csv')
    models = []
    try:
        with open(csv_path, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                row.pop(None, None)
                clean = {k.strip(): v for k, v in row.items()}
                models.append(clean)
    except FileNotFoundError:
        return jsonify({"status": "error", "message": "data.csv not found"}), 404

    return jsonify({"status": "success", "models": models})


@ollama_model_bp.route('/api/ollama/models/installed', methods=['POST'])
def installed_models():
    """Return installed Ollama models, optionally filtered by a model_id prefix."""
    data = request.json or {}
    model_id = data.get('model_id')  # optional filter

    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True, text=True, check=True
        )
        lines = result.stdout.strip().splitlines()

        installed = []
        for i, line in enumerate(lines):
            if i == 0 and ("MODEL" in line.upper() or "NAME" in line.upper()):
                continue
            parts = line.split()
            if parts:
                name = parts[0]
                if not model_id or name.startswith(model_id):
                    installed.append(name)

        return jsonify({"status": "success", "installed_models": installed})
    except subprocess.CalledProcessError as e:
        current_app.logger.error(f"Ollama list failed: {e.stderr}")
        return jsonify({
            "status": "error",
            "message": "Failed to list Ollama models"
        }), 500


@ollama_model_bp.route('/api/ollama/models/download', methods=['POST'])
def download_model():
    """Pull an Ollama model locally using `ollama pull` in background."""
    data = request.json or {}
    model_id = data.get('model_id')
    if not model_id:
        return jsonify({
            "status": "error",
            "message": "model_id is required"
        }), 400

    def run_pull():
        download_status[model_id] = 0.0
        proc = subprocess.Popen(
            ["ollama", "pull", model_id],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )
        download_procs[model_id] = proc
        for line in proc.stdout or []:
            m = re.search(r'(\d+(?:\.\d+)?)\s*%', line)
            if m:
                p = float(m.group(1)) / 100.0
                download_status[model_id] = min(max(p, 0.0), 1.0)
        proc.wait()
        download_status[model_id] = 1.0
        download_procs.pop(model_id, None)

    threading.Thread(target=run_pull, daemon=True).start()
    return jsonify({
        "status": "success",
        "message": "Download started"
    }), 202


@ollama_model_bp.route('/api/ollama/models/status/<model_id>', methods=['GET'])
def model_download_status(model_id):
    """Return current download progress for model_id as a float 0.0-1.0, or null if unknown."""
    prog = download_status.get(model_id)
    return jsonify({
        "status": "success",
        "progress": prog  
    }), 200


@ollama_model_bp.route('/api/ollama/models/cancel/<model_id>', methods=['DELETE'])
def cancel_download(model_id):
    """Cancel an ongoing Ollama model pull."""
    proc = download_procs.get(model_id)
    if not proc:
        return jsonify({
            "status": "error",
            "message": f"No download in progress for '{model_id}'"
        }), 404

    try:
        proc.terminate()  
        proc.wait(timeout=5)
    except Exception:
        proc.kill()
    finally:
        download_procs.pop(model_id, None)
        download_status.pop(model_id, None)

    return jsonify({
        "status": "success",
        "message": f"Download of '{model_id}' cancelled"
    }), 200

@ollama_model_bp.route('/api/ollama/models/<model_id>', methods=['DELETE'])
def delete_model(model_id):
    """Remove an Ollama model locally using `ollama remove <model_id>`."""
    try:
        subprocess.run(
            ["ollama", "rm", model_id],
            capture_output=True, text=True, check=True
        )
        return jsonify({
            "status": "success",
            "message": f"Model '{model_id}' removed"
        }), 200

    except subprocess.CalledProcessError as e:
        current_app.logger.error(f"Ollama remove failed: {e.stderr}")
        return jsonify({
            "status": "error",
            "message": f"Failed to remove model '{model_id}'"
        }), 500