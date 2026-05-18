from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
import os

app = Flask(__name__)

# =====================================================
# LOAD MODELS SAFELY
# =====================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODELS_DIR = os.path.join(BASE_DIR, '..', 'models')

# Linear Regression
lr_model = joblib.load(
    os.path.join(MODELS_DIR, 'linear_regression.pkl')
)

# Scaler
scaler = joblib.load(
    os.path.join(MODELS_DIR, 'scaler.pkl')
)

# ANN MODEL
ann_model = load_model(
    os.path.join(MODELS_DIR, 'ann_model.h5'),
    compile=False
)

# =====================================================
# DASHBOARD
# =====================================================

@app.route('/')
def dashboard():
    return render_template('dashboard.html')

# =====================================================
# ANALYTICS
# =====================================================

@app.route('/analytics')
def analytics():
    return render_template('analytics.html')

# =====================================================
# ABOUT
# =====================================================

@app.route('/about')
def about():
    return render_template('about.html')

# =====================================================
# PREDICTION
# =====================================================

@app.route('/prediction', methods=['GET', 'POST'])
def prediction():

    prediction_result = None
    error_message = None

    if request.method == 'POST':

        try:

            year = float(request.form['year'])
            feature1 = float(request.form['feature1'])
            feature2 = float(request.form['feature2'])
            feature3 = float(request.form['feature3'])

            input_data = np.array([
                [year, feature1, feature2, feature3]
            ])

            input_scaled = scaler.transform(input_data)

            prediction_result = ann_model.predict(
                input_scaled
            )[0][0]

            prediction_result = round(
                float(prediction_result), 4
            )

        except Exception as e:

            error_message = str(e)

    return render_template(
        'prediction.html',
        prediction=prediction_result,
        error=error_message
    )

# =====================================================
# RUN SERVER
# =====================================================

if __name__ == '__main__':

    PORT = int(os.environ.get('PORT', 5000))

    app.run(
        host='0.0.0.0',
        port=PORT,
        debug=True
    )
