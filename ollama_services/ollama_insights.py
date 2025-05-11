import os
import json
import requests
from flask import Blueprint, jsonify, request
import traceback

insights_api = Blueprint('insights_api', __name__)

OLLAMA_BASE_URL = "http://localhost:11434/api"

@insights_api.route('/api/chat/llama', methods=['POST'])
def chat_with_gemma():
    """
    Chat with the Gemma model using the Ollama API.

    Request body:
    {
        "prompt": "Your question or message for Gemma"
    }

    Returns:
        JSON with the model's response.
    """
    data = request.get_json()
    if not data or 'prompt' not in data:
        return jsonify({'error': 'Prompt is required'}), 400

    prompt = data['prompt']

    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/generate",
            json={
                "model": "llama3.2:3b",
                "prompt": prompt,
                "stream": False
            }
        )

        if response.status_code == 200:
            return jsonify(response.json())
        else:
            return jsonify({'error': f'Ollama API error: {response.status_code}', 'message': response.text}), 500

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500
