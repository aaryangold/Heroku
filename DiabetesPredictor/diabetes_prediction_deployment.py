#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  6 21:08:11 2021

@author: asaap
"""

import numpy as np
import pandas as pd
import pickle


df = pd.read_csv('diabetes.csv')
df = df.rename(columns={'BloodPressure':'BP', 'DiabetesPedigreeFunction':'DPF'})
df[['Glucose', 'BP', 'SkinThickness', 'Insulin', 'BMI', 'DPF']] = df[['Glucose', 'BP', 'SkinThickness', 'Insulin', 'BMI', 'DPF']].replace(0, np.NaN)

df['Glucose'].fillna(df['Glucose'].mean(), inplace=True)
df['BP'].fillna(df['BP'].mean(), inplace=True)
df['SkinThickness'].fillna(df['SkinThickness'].median(), inplace=True)
df['Insulin'].fillna(df['Insulin'].median(), inplace=True)
df['BMI'].fillna(df['BMI'].median(), inplace=True)

from sklearn.model_selection import train_test_split
X = df.drop(columns='Outcome')
y = df.Outcome
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)


from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=20, random_state=0)
classifier.fit(X_train, y_train)

filename = 'diabetes-prediction-model.pkl'
pickle.dump(classifier, open(filename, 'wb'))