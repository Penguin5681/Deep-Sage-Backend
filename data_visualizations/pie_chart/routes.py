import io
import json

import pandas as pd
import matplotlib.pyplot as plt
from flask import request, send_file, jsonify

from . import pie_chart_bp
from .utils import validate_and_parse_input, generate_pie_chart

@pie_chart_bp.route('/api/visualization/generate_pie_chart', methods=['POST'])
def generate_pie_chart_route():
    try:
        if 'csv_file' not in request.files:
            return jsonify({'error': 'CSV file is required'}), 400

        csv_file = request.files['csv_file']
        config = request.form.get('config')

        if not config:
            return jsonify({'error': 'Config JSON is missing'}), 400

        config = json.loads(config) if isinstance(config, str) else config

        df = pd.read_csv(csv_file)
        parsed_df, config = validate_and_parse_input(df, config)

        # Generate and save pie chart
        saved_file_path = generate_pie_chart(parsed_df, config)

        print(saved_file_path)
        return jsonify({"generated_pie_chart": saved_file_path}), 200
        # uncommenting this would straight up return the binary file to an api client
        # the terminal gets fucked opening binary files
        # return send_file(saved_file_path, mimetype=f'image/{config.get("format", "png")}')
    except Exception as e:
        return jsonify({'error': str(e)}), 500
