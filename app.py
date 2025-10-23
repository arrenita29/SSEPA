from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load your trained model (you'll need to upload your .pkl file)
try:
    model = joblib.load('iphone_price_model.pkl')
    print("✅ Model loaded successfully!")
except:
    print("⚠️  Model file not found. Using mock predictions.")
    model = None

# Feature mapping based on your dataset
MODEL_MAPPING = {
    'iphone_se_2': 'iphone se 2',
    'iphone_x': 'iphone x', 
    'iphone_8': 'iphone 8',
    'iphone_11': 'iphone 11',
    'iphone_12': 'iphone 12'
}

SCREEN_DAMAGE_MAPPING = {
    'undamaged': 'undamaged',
    'minor_scratches': 'minor_scratches', 
    'cracked': 'cracked'
}

@app.route('/')
def home():
    return jsonify({
        "message": "Smart Secondhand Electronics Price Advisor API",
        "status": "running",
        "endpoints": {
            "/predict": "POST - Get price prediction",
            "/health": "GET - API health check"
        }
    })

@app.route('/health')
def health():
    return jsonify({"status": "healthy", "timestamp": datetime.now().isoformat()})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from request
        data = request.get_json()
        
        # Extract features
        model_name = data.get('model')
        storage = int(data.get('storage'))
        battery_health = int(data.get('battery_health'))
        screen_condition = data.get('screen_condition')
        months_since_release = int(data.get('months_since_release'))
        
        print(f"📱 Received prediction request: {model_name}, {storage}GB, {battery_health}%")
        
        # Create feature array matching your training data
        features = {
            'Model': MODEL_MAPPING.get(model_name, 'iphone 8'),
            'battery_health': battery_health,
            'battery_renew': False,
            'screen_replacement': False,
            'display_replacement': False,
            'storage': storage,
            'colour': 'Space Gray',  # Default value
            'backglass_damages': False,
            'screen_damages': SCREEN_DAMAGE_MAPPING.get(screen_condition, 'undamaged'),
            'availability': 'full-set',
            'ios_updates': True,
            'Months_since_release': months_since_release,
            'Exchange_rate_1_USD_to_LKR': 341.85,  # From your dataset
            'release_year': 2020,  # Default, you might want to map this per model
            'release_month': 1,    # Default
            'price_year': 2024,    # Current year
            'price_month': 1       # Current month
        }
        
        # Convert to DataFrame
        features_df = pd.DataFrame([features])
        
        if model:
            # Make prediction using your actual model
            prediction_log = model.predict(features_df)[0]
            predicted_price_lkr = np.expm1(prediction_log)  # Reverse log transform
        else:
            # Mock prediction (fallback)
            predicted_price_lkr = mock_predict_price(model_name, storage, battery_health, screen_condition, months_since_release)
        
        # Calculate confidence based on feature quality
        confidence = calculate_confidence(battery_health, screen_condition, months_since_release)
        
        response = {
            'predicted_price': round(predicted_price_lkr),
            'currency': 'LKR',
            'confidence': confidence,
            'features_used': {
                'model': model_name,
                'storage_gb': storage,
                'battery_health_percent': battery_health,
                'screen_condition': screen_condition,
                'device_age_months': months_since_release
            },
            'feature_importance': {
                'model': get_model_impact(model_name),
                'storage': get_storage_impact(storage),
                'battery_health': get_battery_impact(battery_health),
                'screen_condition': get_screen_impact(screen_condition),
                'age': get_age_impact(months_since_release)
            }
        }
        
        return jsonify(response)
        
    except Exception as e:
        print(f"❌ Prediction error: {str(e)}")
        return jsonify({'error': str(e)}), 400

def mock_predict_price(model, storage, battery_health, screen_condition, months_since_release):
    """Mock prediction function - replace with your actual model"""
    base_prices = {
        'iphone_se_2': 60000,
        'iphone_x': 75000,
        'iphone_8': 45000,
        'iphone_11': 85000,
        'iphone_12': 95000
    }
    
    price = base_prices.get(model, 50000)
    
    # Adjust for storage
    price += (storage / 64 - 1) * 15000
    
    # Adjust for battery health
    price *= (battery_health / 100)
    
    # Adjust for screen condition
    screen_multipliers = {
        'undamaged': 1.0,
        'minor_scratches': 0.85,
        'cracked': 0.6
    }
    price *= screen_multipliers.get(screen_condition, 0.7)
    
    # Adjust for age
    price *= max(0.3, 1 - (months_since_release * 0.008))
    
    return round(price)

def calculate_confidence(battery_health, screen_condition, age_months):
    """Calculate prediction confidence based on input quality"""
    battery_score = min(1.0, battery_health / 100)
    screen_score = 1.0 if screen_condition == 'undamaged' else 0.8
    age_score = max(0.5, 1 - (age_months / 120))
    
    confidence = (battery_score + screen_score + age_score) / 3
    return round(confidence * 100, 1)

def get_model_impact(model):
    impacts = {
        'iphone_se_2': 35,
        'iphone_x': 40,
        'iphone_8': 30,
        'iphone_11': 45,
        'iphone_12': 50
    }
    return impacts.get(model, 35)

def get_storage_impact(storage):
    return min(25, (storage / 64) * 10)

def get_battery_impact(battery_health):
    return max(5, (100 - battery_health) / 2)

def get_screen_impact(screen_condition):
    impacts = {
        'undamaged': 5,
        'minor_scratches': 15,
        'cracked': 25
    }
    return impacts.get(screen_condition, 10)

def get_age_impact(months):
    return min(20, months / 6)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
