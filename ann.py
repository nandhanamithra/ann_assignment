import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

data=pd.read_csv(r"C:\flutter_projects\INTERNSHIP\lvl 2\class\deeplearning\heart.csv")

#print(data.head())
#print(data.columns)
encoder = LabelEncoder()
data['thal'] = encoder.fit_transform(data['thal'])
x = data.drop("target", axis=1)
y = data["target"]
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x_scaled,y,test_size=0.2,random_state=42
)

model = Sequential([
    Dense(16, activation='relu', input_shape=(13,)),
    Dense(8, activation='relu'),
    Dense(4, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
model.fit(x_train,y_train,epochs=100)
loss, accuracy = model.evaluate(x_test, y_test)
print("Accuracy:", accuracy)

joblib.dump(model,"heart_model.pkl")
joblib.dump(scaler, "heart_scaler.pkl")
joblib.dump(encoder,"heart_encoder.pkl")
print("Model, scaler, and encoder saved successfully.")