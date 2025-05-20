# -*- coding: utf-8 -*-
"""
Created on Thu May 30 17:19:12 2024

@author: user
"""
from flask import Flask, request, render_template
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, cross_val_predict
from sklearn.metrics import precision_recall_curve
import lightgbm as lgb
import requests
import io
import random
import os

app = Flask(__name__)

# Set global random seed for reproducibility
random_state = 42
random.seed(random_state)
np.random.seed(random_state)

def calculate_f1_optimal_threshold(X, y, model):
    """Calculate the optimal threshold based on F1-score using cross-validation."""
    # Use cross-validation to get predicted probabilities
    y_proba = cross_val_predict(model, X, y, cv=5, method='predict_proba')[:, 1]
    
    # Calculate precision, recall, and thresholds
    precision, recall, thresholds = precision_recall_curve(y, y_proba)
    
    # Compute F1-scores for each threshold
    f1_scores = 2 * (precision * recall) / (precision + recall)
    f1_scores = np.nan_to_num(f1_scores)  # Replace NaN with 0
    
    # Find the threshold that maximizes F1-score
    best_threshold = thresholds[np.argmax(f1_scores)]
    return best_threshold

def train_model():
    url = 'https://raw.githubusercontent.com/xyf19912015/myapp-flask3/main/KDSS31.csv'
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Check if the request was successful

        content = response.content.decode('utf-8')

        # Check if content is HTML instead of CSV
        if '<!DOCTYPE html>' in content or '<html>' in content:
            print("The content retrieved is an HTML page, not a CSV file.")
            return None, None, None, None

        data = pd.read_csv(io.StringIO(content), encoding='gbk')

        # Prepare features and target
        X = data[['T cell%', 'NLR', 'IL-6', 'CRP', 'NT-proBNP']]
        y = data['KDSS']

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Calculate scale_pos_weight for class imbalance
        neg_count = (y == 0).sum()
        pos_count = (y == 1).sum()
        scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1

        # Initialize LightGBM classifier with scale_pos_weight
        lgb_classifier = lgb.LGBMClassifier(random_state=random_state, scale_pos_weight=scale_pos_weight)

        # Define hyperparameter grid
        param_grid = {
            'n_estimators': [200],
            'learning_rate': [0.05],
            'max_depth': [7],
            'num_leaves': [30],
            'min_child_samples': [10],
            'subsample': [0.6],
            'colsample_bytree': [0.6],
            'reg_lambda': [0.1],
            'reg_alpha': [1.0]
        }

        # Perform grid search
        grid_search = GridSearchCV(estimator=lgb_classifier, param_grid=param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
        grid_search.fit(X_scaled, y)

        best_lgb = grid_search.best_estimator_

        # Calculate optimal threshold based on F1-score
        best_threshold = calculate_f1_optimal_threshold(X_scaled, y, best_lgb)

        return scaler, best_lgb, X.columns, best_threshold
    except requests.exceptions.HTTPError as e:
        print(f"HTTP error occurred: {e}")
        return None, None, None, None
    except pd.errors.ParserError as e:
        print(f"Parser error: {e}")
        return None, None, None, None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None, None, None, None

# Train the model and get the parameters
scaler, best_lgb, feature_names, best_threshold = train_model()

@app.route('/')
def home():
    annotations = {
        'T cell%': 'T cell percentage, %',
        'NLR': 'Neutrophil to Lymphocyte Ratio',
        'IL-6': 'Interleukin-6 level, pg/mL',
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

    prediction_proba = best_lgb.predict_proba(input_scaled)[:, 1][0]
    risk_level = "High Risk!" if prediction_proba > best_threshold else "Low Risk!"
    risk_color = "red" if prediction_proba > best_threshold else "green"
    prediction_rounded = round(prediction_proba, 4)

    return render_template('result.html', prediction=prediction_rounded, best_threshold=best_threshold, risk_level=risk_level, risk_color=risk_color)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)), debug=True)
