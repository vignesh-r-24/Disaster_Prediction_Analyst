import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import joblib

# Load your saved model
# model = joblib.load("/content/disaster_model.pkl")

df = pd.read_csv('disasters.csv')

feature_columns = ['Magnitude','Fatalities']
target_column = 'Disaster_Type'

X = df[feature_columns]
y = df[target_column]

le = LabelEncoder()
y = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

joblib.dump(model, 'disaster_model.pkl')
joblib.dump(le, 'label_encoder.pkl')

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

model = joblib.load('disaster_model.pkl')
le = joblib.load('label_encoder.pkl')

st.title(" Disaster Type Prediction ")

# User inputs
st.header("Enter Disaster Details:")
mag = st.number_input("Magnitude", min_value=0.0, value=5.0, step=0.1)
fat = st.number_input("Fatalities", min_value=0, value=100, step=10)
# eco = st.number_input("Economic Loss ($)", min_value=0, value=1000000, step=1000)

# Predict button
if st.button("Predict Disaster Type"):
    pred = model.predict([[mag, fat]])
    disaster = le.inverse_transform(pred)[0]
    st.success(f"Predicted Disaster Type: **{disaster}**")