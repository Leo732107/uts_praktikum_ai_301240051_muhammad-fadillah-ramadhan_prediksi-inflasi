from flask import Flask, render_template

app = Flask(__name__)

# =========================
# HOME
# =========================
@app.route('/')
def home():
    return render_template('dashboard.html')

# =========================
# DASHBOARD
# =========================
@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

# =========================
# PREDICTION
# =========================
@app.route('/prediction')
def prediction():
    return render_template('prediction.html')

# =========================
# ANALYTICS
# =========================
@app.route('/analytics')
def analytics():
    return render_template('analytics.html')

# =========================
# ABOUT
# =========================
@app.route('/about')
def about():
    return render_template('about.html')

# =========================
# MODELS
# =========================
@app.route('/models')
def models():
    return render_template('models.html')

# =========================
# RUN
# =========================
if __name__ == '__main__':
    app.run(debug=True)