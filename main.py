import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load models
model_heel = joblib.load(r'model_rf.joblib')
model_soft = joblib.load(r'model_rf_Soft.joblib')
model_tread = joblib.load(r'model_rf_thread.joblib')

# Function to convert duration in HH:MM:SS format to seconds
def duration_to_seconds(duration):
    h, m, s = map(int, duration.split(':'))
    return h * 3600 + m * 60 + s

def preprocess_heel_input(tyre_size, compound, act_tem, dur_start_to_measured, no_of_layers):
    input_data = {
        'Heel_Act_Tem.': act_tem,
        'Heel_Dur_Start_to_Measured_2': duration_to_seconds(dur_start_to_measured),
        'Heel_No_of_layers': no_of_layers,
    }
    # One-hot encoding for tyre_size
    for size in ['140/55-9', '18X7-8', '2.00/50-10', '5.00-8', '6.00-9', '6.50-10', '7.00-12']:
        input_data[f'Tyre size_{size}'] = tyre_size == size
    # One-hot encoding for compound
    input_data['Heel_Compound_1121'] = compound == '1121'
    
    return pd.DataFrame([input_data])

def preprocess_soft_input(tyre_size, compound, act_tem, dur_start_to_measured, no_of_layers):
    input_data = {
        'Soft_Act_Tem': act_tem,
        'Soft_Dur_Start_to_Measured_2': duration_to_seconds(dur_start_to_measured),
        'Soft_No_of_layers': no_of_layers,
    }
    # One-hot encoding for tyre_size
    for size in ['140/55-9', '18X7-8', '2.00/50-10', '5.00-8', '6.00-9', '6.50-10', '7.00-12']:
        input_data[f'Tyre size_{size}'] = tyre_size == size
    # One-hot encoding for compound
    input_data['Soft_Compound_03C090'] = compound == '03C090'
    
    return pd.DataFrame([input_data])

def preprocess_tread_input(tyre_size, compound, act_tem, dur_start_to_press, no_of_layers):
    input_data = {
        'Thread_Act_Tem': act_tem,
        'Thread_Dur_Start_to_press': duration_to_seconds(dur_start_to_press),
        'Thread_No_of_layers': no_of_layers,
    }
    # One-hot encoding for tyre_size
    for size in ['140/55-9', '18X7-8', '2.00/50-10', '5.00-8', '6.00-9', '6.50-10', '7.00-12']:
        input_data[f'Tyre size_{size}'] = tyre_size == size
    # One-hot encoding for compound
    input_data['Thread_Compound_1110'] = compound == '1110'
    
    return pd.DataFrame([input_data])

# Streamlit app
st.title("Tyre Temperature Prediction")

st.header("Predict Heel Final Temperature")
tyre_size_heel = st.selectbox("Tyre Size (Heel)", ['140/55-9', '18X7-8', '2.00/50-10', '5.00-8', '6.00-9', '6.50-10', '7.00-12'])
heel_compound = st.selectbox("Heel Compound", ['1121'])
heel_act_tem = st.number_input("Heel Actual Temperature", step=0.1)
heel_dur_start_to_measured = st.text_input("Heel Duration Start to Measured (HH:MM:SS)")
heel_no_of_layers = st.number_input("Heel Number of Layers", step=0.1)

if st.button("Predict Heel Final Temperature"):
    heel_input = preprocess_heel_input(tyre_size_heel, heel_compound, heel_act_tem, heel_dur_start_to_measured, heel_no_of_layers)
    heel_final_tem = model_heel.predict(heel_input)[0]
    st.write(f"Predicted Heel Final Temperature: {heel_final_tem}")

st.header("Predict Soft Final Temperature")
tyre_size_soft = st.selectbox("Tyre Size (Soft)", ['140/55-9', '18X7-8', '2.00/50-10', '5.00-8', '6.00-9', '6.50-10', '7.00-12'])
soft_compound = st.selectbox("Soft Compound", ['03C090'])
soft_act_tem = st.number_input("Soft Actual Temperature", step=0.1)
soft_dur_start_to_measured = st.text_input("Soft Duration Start to Measured (HH:MM:SS)")
soft_no_of_layers = st.number_input("Soft Number of Layers", step=0.1)

if st.button("Predict Soft Final Temperature"):
    soft_input = preprocess_soft_input(tyre_size_soft, soft_compound, soft_act_tem, soft_dur_start_to_measured, soft_no_of_layers)
    soft_final_tem = model_soft.predict(soft_input)[0]
    st.write(f"Predicted Soft Final Temperature: {soft_final_tem}")

st.header("Predict Tread Final Temperature")
tyre_size_tread = st.selectbox("Tyre Size (Tread)", ['140/55-9', '18X7-8', '2.00/50-10', '5.00-8', '6.00-9', '6.50-10', '7.00-12'])
tread_compound = st.selectbox("Tread Compound", ['1110'])
tread_act_tem = st.number_input("Tread Actual Temperature", step=0.1)
tread_dur_start_to_press = st.text_input("Tread Duration Start to Press (HH:MM:SS)")
tread_no_of_layers = st.number_input("Tread Number of Layers", step=0.1)

if st.button("Predict Tread Final Temperature"):
    tread_input = preprocess_tread_input(tyre_size_tread, tread_compound, tread_act_tem, tread_dur_start_to_press, tread_no_of_layers)
    tread_final_tem = model_tread.predict(tread_input)[0]
    st.write(f"Predicted Tread Final Temperature: {tread_final_tem}")
