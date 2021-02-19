#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 12:15:34 2021

@author: asaap
"""

import numpy as np
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('diabetes-prediction-model.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        pregnancies = int(request.form['pregnancies'])
        glucose = int(request.form['glucose'])
        bp = int(request.form['bloodpressure'])
        skinthickness = int(request.form['skinthickness'])
        insulin = int(request.form['insulin'])
        bmi = float(request.form['bmi'])
        dpf = float(request.form['dpf'])
        age = int(request.form['age'])
        
        data = np.array([pregnancies, glucose, bp, skinthickness, insulin, bmi, dpf, age])
        prediction = model.predict(data)
        
        if prediction == 1:
            pred = "SORRY! You have DIABETES."
        elif prediction == 0:
            pred = "VOILA! You don't have Diabetes."
        
        
        output = pred
    

    return render_template('index.html', prediction_text='{}'.format(output))
    
    
if __name__ == "__main__":
    app.run(debug=True)