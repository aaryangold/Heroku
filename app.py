#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 20 11:21:40 2021

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
        bp = int(request.form['bp'])
        skinthickness = int(request.form['skinthickness'])
        insulin = int(request.form['insulin'])
        bmi = float(request.form['bmi'])
        dpf = float(request.form['dpf'])
        age = int(request.form['age'])
        
    data = np.array([[pregnancies, glucose, bp, skinthickness, insulin, bmi, dpf, age]])
    prediction = model.predict(data)
        

    return render_template('result.html', prediction=prediction)
    
    
if __name__ == "__main__":
    app.run(debug=True)