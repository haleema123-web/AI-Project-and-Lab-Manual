from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import joblib
import pandas as pd

app = Flask(__name__)
CORS(app)

# Load trained ML model
model = joblib.load("car_price_model.pkl")

# Label mappings (replace with your actual LabelEncoder mappings)
brand_mapping = {"Maruti": 0, "Hyundai": 1, "Honda": 2}
model_mapping = {"Swift": 0, "i20": 1, "City": 2}
fuel_mapping = {"Petrol": 0, "Diesel": 1, "CNG": 2}
transmission_mapping = {"Manual": 0, "Automatic": 1}

# Serve index.html
@app.route('/')
def home():
    return send_from_directory('.', 'index.html')

# Predict route
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    try:
        brand = brand_mapping.get(data.get("brand"), 0)
        model_name = model_mapping.get(data.get("model"), 0)
        fuel = fuel_mapping.get(data.get("fuel_type"), 0)
        transmission = transmission_mapping.get(data.get("transmission"), 0)

        input_df = pd.DataFrame([{
            "brand": brand,
            "model": model_name,
            "year": data["year"],
            "mileage": data["mileage"],
            "engine_size": data["engine_size"],
            "fuel_type": fuel,
            "transmission": transmission
        }])

        predicted_price = model.predict(input_df)[0]
        return jsonify({"predicted_price": round(predicted_price, 2)})

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
