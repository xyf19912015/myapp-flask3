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
import lightgbm as lgb  # 导入 LightGBM 包
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
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # 检查请求是否成功

        content = response.content.decode('utf-8')

        # 检查内容是否为HTML
        if '<!DOCTYPE html>' in content or '<html>' in content:
            print("The content retrieved is an HTML page, not a CSV file.")
            return None, None, None, None, None

        data = pd.read_csv(io.StringIO(content), encoding='gbk')

        # 训练代码
        X = data[['THLC', 'NLR', 'GLB', 'WBC', 'CRP', 'NT-proBNP']]
        y = data['KDSS']

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        smote = SMOTE(sampling_strategy=0.5, random_state=random_state)
        X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

        # 使用 LightGBM 
        lgb_classifier = lgb.LGBMClassifier(random_state=random_state)

        # 更新超参数设置
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

        grid_search = GridSearchCV(estimator=lgb_classifier, param_grid=param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
        grid_search.fit(X_resampled, y_resampled)

        best_lgb = grid_search.best_estimator_

        best_threshold, best_youden_index = cross_validated_youden_index(X_resampled, y_resampled, best_lgb)

        return scaler, best_lgb, X.columns, best_threshold, best_youden_index
    except requests.exceptions.HTTPError as e:
        print(f"HTTP error occurred: {e}")
        return None, None, None, None, None
    except pd.errors.ParserError as e:
        print(f"Parser error: {e}")
        return None, None, None, None, None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None, None, None, None, None

# Train the model and get the parameters
scaler, best_lgb, feature_names, best_threshold, best_youden_index = train_model()

@app.route('/')
def home():
    annotations = {
        'THLC': 'Total Helper Lymphocyte Count,/mL',
        'NLR': 'Neutrophil to Lymphocyte Ratio',
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

    prediction_proba = best_lgb.predict_proba(input_scaled)[:, 1][0]
    risk_level = "High Risk!" if prediction_proba > best_threshold else "Low Risk!"
    risk_color = "red" if prediction_proba > best_threshold else "green"
    prediction_rounded = round(prediction_proba, 4)

    return render_template('result.html', prediction=prediction_rounded, youden_index=best_youden_index, best_threshold=best_threshold, risk_level=risk_level, risk_color=risk_color)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)), debug=True)
