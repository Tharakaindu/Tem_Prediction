import streamlit as st
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go

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

# Load models
model_heel = joblib.load(r'model_rf.joblib')
model_soft = joblib.load(r'model_rf_Soft.joblib')
model_tread = joblib.load(r'model_rf_thread.joblib')

# Set page configuration
st.set_page_config(page_title="Tyre Curing Cycle Temperature Prediction", layout="wide", initial_sidebar_state="expanded")

# Streamlit app
st.title("Tyre Curing Cycle Temperature Prediction")

# Load background image
background_image = st.image(r'C:\Users\jfernand3\Documents\temparature\2-slide-proxima.png', use_column_width=True)

# Define columns layout
col1, col2, col3 = st.columns(3)

# Inputs for Heel
with col1:
    st.header("Heel")
    tyre_size_heel = st.selectbox("Tyre Size (Heel)", ['140/55-9', '18X7-8', '2.00/50-10', '5.00-8', '6.00-9', '6.50-10', '7.00-12'], key="heel_tyre_size")
    heel_compound = st.selectbox("Compound (Heel)", ['1121'], key="heel_compound")
    heel_act_tem = st.number_input("Temperature (Heel)", step=0.1, key="heel_act_tem")
    heel_dur_start_to_measured = st.text_input("Duration Rolling to Press (HH:MM:SS) (Heel)", key="heel_dur_start")
    heel_no_of_layers = st.number_input("Number of Layers (Heel)", step=0.1, key="heel_layers")

# Inputs for Soft
with col2:
    st.header("Soft")
    tyre_size_soft = st.selectbox("Tyre Size (Soft)", ['140/55-9', '18X7-8', '2.00/50-10', '5.00-8', '6.00-9', '6.50-10', '7.00-12'], key="soft_tyre_size")
    soft_compound = st.selectbox("Compound (Soft)", ['03C090'], key="soft_compound")
    soft_act_tem = st.number_input("Actual Temperature (Soft)", step=0.1, key="soft_act_tem")
    soft_dur_start_to_measured = st.text_input("Duration Rolling to Press (HH:MM:SS) (Soft)", key="soft_dur_start")
    soft_no_of_layers = st.number_input("Number of Layers (Soft)", step=0.1, key="soft_layers")

# Inputs for Tread
with col3:
    st.header("Tread")
    tyre_size_tread = st.selectbox("Tyre Size (Tread)", ['140/55-9', '18X7-8', '2.00/50-10', '5.00-8', '6.00-9', '6.50-10', '7.00-12'], key="tread_tyre_size")
    tread_compound = st.selectbox("Compound (Tread)", ['1110'], key="tread_compound")
    tread_act_tem = st.number_input("Actual Temperature (Tread)", step=0.1, key="tread_act_tem")
    tread_dur_start_to_press = st.text_input("Duration Rolling to Press (HH:MM:SS) (Tread)", key="tread_dur_start")
    tread_no_of_layers = st.number_input("Number of Layers (Tread)", step=0.1, key="tread_layers")

# Predict button
if st.button("Predict Temperatures"):
    if not all([tyre_size_heel, heel_compound, heel_act_tem, heel_dur_start_to_measured, heel_no_of_layers,
                tyre_size_soft, soft_compound, soft_act_tem, soft_dur_start_to_measured, soft_no_of_layers,
                tyre_size_tread, tread_compound, tread_act_tem, tread_dur_start_to_press, tread_no_of_layers]):
        st.error("Please fill in all the required fields for all sections (Heel, Soft, Tread) before predicting.")
    else:
        heel_input = preprocess_heel_input(tyre_size_heel, heel_compound, heel_act_tem, heel_dur_start_to_measured, heel_no_of_layers)
        soft_input = preprocess_soft_input(tyre_size_soft, soft_compound, soft_act_tem, soft_dur_start_to_measured, soft_no_of_layers)
        tread_input = preprocess_tread_input(tyre_size_tread, tread_compound, tread_act_tem, tread_dur_start_to_press, tread_no_of_layers)
        
        heel_final_tem = model_heel.predict(heel_input)[0]
        soft_final_tem = model_soft.predict(soft_input)[0]
        tread_final_tem = model_tread.predict(tread_input)[0]
        
        # Calculate differences
        heel_difference = heel_final_tem - heel_act_tem
        soft_difference = soft_final_tem - soft_act_tem
        tread_difference = tread_final_tem - tread_act_tem
        
        # Display results in a table
        results = pd.DataFrame({
            'Component': ['Heel', 'Soft', 'Tread'],
            'Actual Temperature': [heel_act_tem, soft_act_tem, tread_act_tem],
            'Predicted Temperature': [heel_final_tem.round(1), soft_final_tem.round(1), tread_final_tem.round(1)],
            'Difference': [heel_difference.round(1), soft_difference.round(1), tread_difference.round(1)]
        })

        st.write(results)

        # Plotting the deviation graph using Plotly
        components = ['Heel', 'Soft', 'Tread']
        actual_temps = [heel_act_tem, soft_act_tem, tread_act_tem]
        predicted_temps = [heel_final_tem, soft_final_tem, tread_final_tem]
        deviations = np.array(predicted_temps) - np.array(actual_temps)
        
        fig = go.Figure()
        
        # Add actual temperatures
        fig.add_trace(go.Bar(
            x=components,
            y=actual_temps,
            name='Actual Temperature',
            marker_color='blue'
        ))

        # Add predicted temperatures
        fig.add_trace(go.Bar(
            x=components,
            y=predicted_temps,
            name='Predicted Temperature',
            marker_color='red'
        ))

        # Add deviation lines
        for i, component in enumerate(components):
            deviation_x = [component, component]
            deviation_y = [actual_temps[i], predicted_temps[i]]
            fig.add_trace(go.Scatter(
                x=deviation_x,
                y=deviation_y,
                mode='lines+markers',
                name=f'{component} Deviation',
                line=dict(color='green', width=2),
                marker=dict(symbol='circle', size=10),
                showlegend=False
            ))

        fig.update_layout(
            title='Actual vs Predicted Temperatures with Deviation',
            xaxis_title='Components',
            yaxis_title='Temperature',
            barmode='group',
            legend_title='Legend'
        )

        st.plotly_chart(fig)
