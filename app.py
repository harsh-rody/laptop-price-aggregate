import pickle
import numpy as np
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

model  = pickle.load(open('model.pkl',  'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

# Must match the exact order used during training:
# ['Stars','Ratings','Reviews','MRP','Ram_GB','SSD_GB','HDD_GB',
#  'Processor_Tier','Processor_Gen','Processor_Brand_Intel','Processor_Brand_Qualcomm']

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        brand = data.get('processor_brand', 'AMD')
        is_intel    = 1 if brand == 'Intel'    else 0
        is_qualcomm = 1 if brand == 'Qualcomm' else 0

        features = np.array([[
            float(data['stars']),
            float(data['ratings']),
            float(data['reviews']),
            float(data['mrp']),
            float(data['ram_gb']),
            float(data['ssd_gb']),
            float(data['hdd_gb']),
            float(data['processor_tier']),
            float(data['processor_gen']),
            is_intel,
            is_qualcomm
        ]])

        scaled        = scaler.transform(features)
        predicted     = model.predict(scaled)[0]
        predicted     = max(0, round(predicted, 2))

        return jsonify({
            'success': True,
            'predicted_price': predicted,
            'formatted':       f'₹{predicted:,.0f}'
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


if __name__ == '__main__':
    app.run(debug=False)
