from flask import Flask, render_template, request
import numpy as np
import joblib
from tensorflow.keras.models import load_model

app = Flask(__name__)

# ==============================
# LOAD MODEL & SCALER
# ==============================

lr_model = joblib.load('../models/linear_regression.pkl')
scaler = joblib.load('../models/scaler.pkl')

ann_model = load_model('../models/ann_model.h5')

# ==============================
# DASHBOARD
# ==============================

@app.route('/')
def dashboard():
    return render_template('dashboard.html')

# ==============================
# ANALYTICS PAGE
# ==============================

@app.route('/analytics')
def analytics():
    return render_template('analytics.html')

# ==============================
# ABOUT PAGE
# ==============================

@app.route('/about')
def about():
    return render_template('about.html')

# ==============================
# PREDICTION PAGE
# ==============================

@app.route('/prediction', methods=['GET', 'POST'])
def prediction():

    prediction_result = None

    if request.method == 'POST':

        year = float(request.form['year'])
        feature1 = float(request.form['feature1'])
        feature2 = float(request.form['feature2'])
        feature3 = float(request.form['feature3'])

        input_data = np.array([
            [year, feature1, feature2, feature3]
        ])

        input_scaled = scaler.transform(input_data)

        prediction_result = ann_model.predict(input_scaled)[0][0]

    return render_template(
        'prediction.html',
        prediction=prediction_result
    )

# ==============================
# RUN APP
# ==============================

if __name__ == '__main__':
    app.run(debug=True)
