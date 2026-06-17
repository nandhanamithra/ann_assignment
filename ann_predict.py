import numpy as np
import joblib
import pandas as pd

model = joblib.load("heart_model.pkl")
scaler = joblib.load("heart_scaler.pkl")
encoder = joblib.load("heart_encoder.pkl")

print("Patient Details")

age = float(input("Age: "))
sex = float(input("Sex (1=male, 0=female): "))
cp = float(input("Chest Pain Type (1-4): "))
trestbps = float(input("Resting Blood Pressure: "))
chol = float(input("Cholesterol: "))
fbs = float(input("Fasting Blood Sugar >120 (1=True,0=False): "))
restecg = float(input("Rest ECG (0-2): "))
thalach = float(input("Maximum Heart Rate Achieved: "))
exang = float(input("Exercise Induced Angina (1=yes,0=no): "))
oldpeak = float(input("Oldpeak: "))
slope = float(input("Slope (1-3): "))
ca = float(input("Number of Major Vessels (0-4): "))

thal = input("Thal (fixed / normal / reversible): ")

thal_encoded = encoder.transform([thal])[0]

new_patient = pd.DataFrame(
    [[age, sex, cp, trestbps, chol, fbs, restecg,
      thalach, exang, oldpeak, slope, ca, thal_encoded]],
    columns=['age','sex','cp','trestbps','chol','fbs',
             'restecg','thalach','exang','oldpeak',
             'slope','ca','thal']
)

new_patient_scaled = scaler.transform(new_patient)

prediction = model.predict(new_patient_scaled)[0][0]

print("prediciton")
if prediction >= 0.5:
    print("Heart Disease Detected")
else:
    print("No Heart Disease")
    