import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

# =====================================
# LOAD DATA
# =====================================

df = pd.read_csv('data/inflation_clean.csv')

X = df[
    [
        'CPI',
        'Exchange_Rate',
        'Interest_Rate'
    ]
]

y = df['Inflation']

# =====================================
# SCALING
# =====================================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# SAVE SCALER
joblib.dump(
    scaler,
    'models/scaler.pkl'
)

# =====================================
# SPLIT DATA
# =====================================

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42
)

# =====================================
# LINEAR REGRESSION
# =====================================

linear_model = LinearRegression()

linear_model.fit(X_train, y_train)

joblib.dump(
    linear_model,
    'models/linear_regression.pkl'
)

# =====================================
# ANN MODEL
# =====================================

ann_model = Sequential([
    Dense(
        64,
        activation='relu',
        input_shape=(X_train.shape[1],)
    ),

    Dense(32, activation='relu'),

    Dense(1)
])

ann_model.compile(
    optimizer='adam',
    loss='mse',
    metrics=['mae']
)

ann_model.fit(
    X_train,
    y_train,
    epochs=50,
    batch_size=16,
    validation_split=0.2
)

ann_model.save(
    'models/ann_model.h5'
)

# =====================================
# LSTM MODEL
# =====================================

X_train_lstm = X_train.reshape(
    (X_train.shape[0], 1, X_train.shape[1])
)

X_test_lstm = X_test.reshape(
    (X_test.shape[0], 1, X_test.shape[1])
)

lstm_model = Sequential([
    LSTM(
        64,
        activation='relu',
        input_shape=(1, X_train.shape[1])
    ),

    Dense(32, activation='relu'),

    Dense(1)
])

lstm_model.compile(
    optimizer='adam',
    loss='mse',
    metrics=['mae']
)

lstm_model.fit(
    X_train_lstm,
    y_train,
    epochs=50,
    batch_size=16,
    validation_split=0.2
)

lstm_model.save(
    'models/lstm_model.h5'
)

# =====================================
# KMEANS
# =====================================

kmeans = KMeans(
    n_clusters=3,
    random_state=42
)

kmeans.fit(X_train)

joblib.dump(
    kmeans,
    'models/kmeans.pkl'
)

# =====================================
# BACKPROPAGATION
# =====================================

backprop_model = Sequential([
    Dense(
        128,
        activation='relu',
        input_shape=(X_train.shape[1],)
    ),

    Dense(64, activation='relu'),

    Dense(1)
])

backprop_model.compile(
    optimizer='adam',
    loss='mse',
    metrics=['mae']
)

backprop_model.fit(
    X_train,
    y_train,
    epochs=100,
    batch_size=8,
    validation_split=0.2
)

backprop_model.save(
    'models/backprop_model.h5'
)

print("ALL MODELS TRAINED SUCCESSFULLY")