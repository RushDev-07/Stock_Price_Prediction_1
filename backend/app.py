from flask import Flask, request, jsonify, send_file
import subprocess
import os

app = Flask(__name__)

# Define absolute paths for output files
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
PREDICTION_PLOT_PATH = os.path.join(PROJECT_ROOT, "output", "prediction_plot.png")
CONFUSION_MATRIX_PATH = os.path.join(PROJECT_ROOT, "output", "confusion_matrix.png")
STRATEGY_OUTPUT_PATH = os.path.join(PROJECT_ROOT, "output", "strategy.txt")

@app.route('/run_pipeline', methods=['POST'])
def run_pipeline():
    data = request.get_json()
    ticker = data.get("ticker")

    if not ticker:
        return jsonify({"error": "Ticker is required"}), 400

    # Set the ticker environment variable
    os.environ["TICKER"] = ticker

    try:
        # Run the pipeline script from the root directory
        result = subprocess.run(["python", os.path.join(PROJECT_ROOT, "run_pipeline.py")], check=True, capture_output=True, text=True)
        print("Pipeline output:", result.stdout)

        # Check if the output files were generated
        if os.path.exists(PREDICTION_PLOT_PATH) and os.path.exists(CONFUSION_MATRIX_PATH):
            # Read strategy recommendation from file if it exists
            strategy_recommendation = ""
            if os.path.exists(STRATEGY_OUTPUT_PATH):
                with open(STRATEGY_OUTPUT_PATH, 'r') as file:
                    strategy_recommendation = file.read().strip()

            return jsonify({
                "prediction_plot": "/get_prediction_plot",
                "confusion_matrix": "/get_confusion_matrix",
                "strategy_recommendation": strategy_recommendation
            }), 200
        else:
            return jsonify({"error": "Pipeline did not generate expected output files"}), 500

    except subprocess.CalledProcessError as e:
        print("Pipeline error output:", e.stderr)
        return jsonify({"error": e.stderr}), 500

@app.route('/get_prediction_plot', methods=['GET'])
def get_prediction_plot():
    return send_file(PREDICTION_PLOT_PATH, mimetype='image/png')

@app.route('/get_confusion_matrix', methods=['GET'])
def get_confusion_matrix():
    return send_file(CONFUSION_MATRIX_PATH, mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True)
