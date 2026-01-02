from flask import Flask, request, jsonify
import joblib
import numpy as np
import traceback

app = Flask(__name__)

# Load the model
try:
    model = joblib.load('iris_model.pkl')
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from the POST request
        data = request.get_json(force=True)

        # Check if the 'input' key is in the data
        if 'input' not in data:
            return jsonify({'error': "'input' key is missing in the request data"}), 400

        # Ensure that the 'input' data is in the correct format (a list of numbers)
        input_data = data['input']
        
        if not isinstance(input_data, list):
            return jsonify({'error': "'input' must be a list"}), 400

        # Check that the list contains the correct number of features (for Iris, 4 features)
        if len(input_data) != 4:
            return jsonify({'error': 'Input list must contain 4 features'}), 400

        # Convert input into numpy array and reshape for prediction
        input_array = np.array(input_data).reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(input_array)

        # Return the prediction
        return jsonify({'prediction': int(prediction[0])})

    except Exception as e:
        # Log the stack trace to the console
        print(f"Error during prediction: {e}")
        print(traceback.format_exc())
        return jsonify({'error': 'An error occurred during prediction. Please check the server logs for more details.'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80)
