# -*- coding: utf-8 -*-
"""
Created on Thu May 30 17:19:12 2024

@author: user
"""
from flask import Flask, request, render_template
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
import requests
import io
import random
import os

app = Flask(__name__)

# Set global random seed for reproducibility
random_state = 42
random.seed(random_state)
np.random.seed(random_state)

def calculate_youden_index(y_true, y_proba):
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    youden_index = tpr - fpr
    best_threshold = thresholds[np.argmax(youden_index)]
    best_youden_index = np.max(youden_index)
    return best_threshold, best_youden_index

def cross_validated_youden_index(X, y, model, cv=5):
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    thresholds = []
    youden_indices = []
    
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        model.fit(X_train, y_train)
        y_proba = model.predict_proba(X_test)[:, 1]
        best_threshold, best_youden_index = calculate_youden_index(y_test, y_proba)
        thresholds.append(best_threshold)
        youden_indices.append(best_youden_index)
    
    return np.mean(thresholds), np.mean(youden_indices)

def train_model():
    url = 'https://raw.githubusercontent.com/xyf19912015/myapp-flask3/main/KDSS21.csv'  
    try:
        response = requests.get(url)
        response.raise_for_status()  # Check if the request was successful
        data = pd.read_csv(io.StringIO(response.content.decode('utf-8')), encoding='gbk')

        # Define features and labels
        X = data[['THLC', 'N/L', 'GLB', 'WBC', 'CRP', 'NT-proBNP']]  # Your features
        y = data['KDSS']  # Your target variable

        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Apply SMOTE
        smote = SMOTE(sampling_strategy=0.5, random_state=random_state)
        X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

        # Train the XGBoost classifier
        xgb_classifier = XGBClassifier(random_state=random_state)

        # Grid search parameters
        param_grid = {
            'n_estimators': [300],
            'learning_rate': [0.05],
            'max_depth': [5],
            'min_child_weight': [1],
            'gamma': [0.1],
            'subsample': [0.6],
            'colsample_bytree': [1.0],
            'reg_lambda': [1.0],
            'reg_alpha': [0.1]
        }

        grid_search = GridSearchCV(estimator=xgb_classifier, param_grid=param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
        grid_search.fit(X_resampled, y_resampled)

        best_xgb = grid_search.best_estimator_

        best_threshold, best_youden_index = cross_validated_youden_index(X_resampled, y_resampled, best_xgb)

        return scaler, best_xgb, X.columns, best_threshold, best_youden_index
    except requests.exceptions.HTTPError as e:
        print(f"HTTP error occurred: {e}")
    except pd.errors.ParserError as e:
        print(f"Parser error: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

# Train the model and get the parameters
scaler, best_xgb, feature_names, best_threshold, best_youden_index = train_model()

@app.route('/')
def home():
    annotations = {
        'THLC': 'Total Helper Lymphocyte Count,/mL',
        'N/L': 'Neutrophil to Lymphocyte Ratio',
        'GLB': 'Globulin level, g/L',
        'WBC': 'White Blood Cell count, 10^9/L',
        'CRP': 'C-Reactive Protein, mg/L',
        'NT-proBNP': 'N-terminal pro B-type Natriuretic Peptide, pg/mL'
    }
    return render_template('index.html', features=annotations.keys(), annotations=annotations)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_features = []
        for feature in feature_names:
            if feature in request.form:
                value = float(request.form[feature])
                input_features.append(value)
            else:
                input_features.append(0.0)  # If feature is missing, fill with 0
    except KeyError as e:
        return f"Error: Missing input for feature: {e.args[0]}"

    input_df = pd.DataFrame([input_features], columns=feature_names)
    input_scaled = scaler.transform(input_df)

    prediction_proba = best_xgb.predict_proba(input_scaled)[:, 1][0]
    risk_level = "High Risk!" if prediction_proba > best_threshold else "Low Risk!"
    risk_color = "red" if prediction_proba > best_threshold else "green"
    prediction_rounded = round(prediction_proba, 4)

    return render_template('result.html', prediction=prediction_rounded, youden_index=best_youden_index, best_threshold=best_threshold, risk_level=risk_level, risk_color=risk_color)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)), debug=True)
