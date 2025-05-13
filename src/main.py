import os
import sys
# DON'T CHANGE THIS !!!
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from flask import Flask, request, jsonify, render_template, send_from_directory
from src.Predict import predict_smiles # Import the prediction function

app = Flask(__name__, 
            static_folder=os.path.join(os.path.dirname(__file__), 'static'),
            template_folder=os.path.join(os.path.dirname(__file__), 'templates'))
app.config['SECRET_KEY'] = 'asdf#FGSgvasgf$5$WGT'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/mof-architect')
def mof_architect_page():
    return render_template('mof_architect.html')

@app.route('/predict', methods=['POST'])
def predict_route():
    try:
        data = request.get_json()
        smiles_string = data.get('smiles')
        if not smiles_string:
            return jsonify({'error': 'SMILES string is required'}), 400
        
        prediction_result = predict_smiles(smiles_string)
        
        # Check if the result from predict_smiles is an error message
        if isinstance(prediction_result, str) and prediction_result.startswith("Prediction failed"):
            return jsonify({'error': prediction_result}), 400 # Return 400 for client-side errors like invalid SMILES
        
        return jsonify({'prediction': prediction_result})
    except Exception as e:
        app.logger.error(f"Prediction error: {e}")
        return jsonify({'error': f"An unexpected server error occurred: {str(e)}"}), 500

@app.route('/<path:path>')
def serve_static(path):
    static_folder_path = app.static_folder
    if static_folder_path is None:
            return "Static folder not configured", 404
    if os.path.exists(os.path.join(static_folder_path, path)):
        return send_from_directory(static_folder_path, path)
    else:
        return "Resource not found", 404

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

