from datetime import datetime
import hopsworks
import joblib
import pandas as pd
import numpy as np
import streamlit as st
from load_model import model, preprocessing_pipeline, heart_fg


def impute(df):
    _, numerical_pipeline, numerical = preprocessing_pipeline.transformers_[0]
    _, categorical_pipeline, categorical = preprocessing_pipeline.transformers_[1]

    numerical_imputer = numerical_pipeline.named_steps['imputer']
    categorical_imputer = categorical_pipeline.named_steps['imputer']

    df[numerical] = numerical_imputer.transform(df[numerical])
    df[categorical] = categorical_imputer.transform(df[categorical])

    return df

def predict(df):
    df = preprocessing_pipeline.transform(df)
    prediction = model.predict(df)
    proba = model.predict_proba(df)
    return prediction[0], proba[0]

def heart(heartdisease, smoking, alcoholdrinking, stroke, diffwalking, gender, agecategory, race, diabetic, physicalactivity, genhealth, asthma, kidneydisease, skincancer, mentalhealth, physicalhealth, sleeptime, bmi):
    df = pd.DataFrame({
        'smoking': [smoking],
        'alcohol_drinking': [alcoholdrinking],
        'stroke': [stroke],
        'diff_walking': [diffwalking],
        'sex': [gender],
        'age_category': [agecategory],
        'race': [race],
        'diabetic': [diabetic],
        'physical_activity': [physicalactivity],
        'gen_health': [genhealth],
        'asthma': [asthma],
        'kidney_disease': [kidneydisease],
        'skin_cancer': [skincancer],
        'b_m_i': [bmi],
        'mental_health': [mentalhealth],
        'physical_health': [physicalhealth],
        'sleep_time': [sleeptime],
    })

    # Replace Unknowns with NaNs
    # Feature pipeline has an imputer
    df = df.replace('Unknown', np.nan)
        
    store_data = False

    if heartdisease != "Unknown":
        df = impute(df)

        df['heart_disease'] = np.float64(heartdisease == "Yes")
        df['timestamp'] = pd.to_datetime(datetime.now())
        # Hacky fix due to Hopsworks Magic
        df["timestamp"] = df['timestamp'] - pd.to_timedelta(0 * df.index, unit='s')
        
        try:
            heart_fg.insert(df, write_options={"wait_for_job": False})
        except Exception as e:
            st.error(f"An error occurred: {e}")

        store_data = True
    
    if store_data:
         st.info("Thank you for submitting your data. We will use it to improve our model.")

    pred, proba = predict(df)
    if not pred:
        return "We predict that you do NOT have heart disease. (But this is not medical advice!)"
    else:
        return "We predict that you MIGHT have heart disease. (But this is not medical advice!)"



# UI Elements
st.title("Heart Disease Prediction")
st.write("Please fill in the following fields to predict if you have heart disease.")

#Defining Some variables
answer1 = ["Yes", "No", "Unknown"]
genanswer = ["Male", "Female", "Unknown"]
agecat = ['18-24', '25-29', '30-34', '35-39', '40-44', '45-49', '50-54', '55-59', '60-64', 
                            '65-69', '70-74', '75-79', '80 or older', 'Unknown']
racecat = ['Unknown', 'American Indian/Alaskan Native', 'Asian', 'Black', 'Hispanic', 'Other', 'White']
diabeticcat = ['Yes', 'No', 'Yes (during pregnancy)', 'Unknown']
rating = ['Excellent', 'Very good', 'Good', 'Fair', 'Poor', 'Unknown']

with st.form(key='heart_form'):
    col1, col2, col3 = st.columns(3)
    with col1:
        heartdisease = st.selectbox("Do you have heart disease?", answer1, key='heartdisease',index=None)
        smoking = st.selectbox("Do you smoke?", answer1, key='smoking', index=None)
        alcoholdrinking = st.selectbox("Do you drink alcohol?", answer1, key='alcoholdrinking',index=None)
        stroke = st.selectbox("Have you had a stroke?", answer1, key='stroke',index=None)
        diffwalking = st.selectbox("Do you have difficulty walking?", answer1, key='diffwalking',index=None)
        gender = st.selectbox("What is your gender?", genanswer, key='gender',index=None)
        agecategory = st.selectbox("What is your age category?", agecat, key='agecategory',index=None)
    with col2:
        race = st.selectbox("What is your race?", racecat, key='race',index=None)
        diabetic = st.selectbox("Are you diabetic?", diabeticcat, key='diabetic',index=None)
        physicalactivity = st.selectbox("Do you engage in physical activity?", answer1, key='physicalactivity',index=None)
        genhealth = st.selectbox("How would you rate your general health?", rating, key='genhealth',index=None)
        asthma = st.selectbox("Do you have asthma?", answer1, key='asthma',index=None)
        kidneydisease = st.selectbox("Do you have kidney disease?", answer1, key='kidneydisease',index=None)
        skincancer = st.selectbox("Do you have skin cancer?", answer1, key='skincancer',index=None)
    with col3:
        mentalhealth = st.slider("How would you rate your mental health?", min_value=0, max_value=30)
        physicalhealth = st.slider("How would you rate your physical health?", min_value=0, max_value=30)
        sleeptime = st.slider("How many hours do you sleep per night?", min_value=0, max_value=24)
        bmi = st.slider("What is your BMI?", min_value=0, max_value=100)

    submit_button = st.form_submit_button(label='Predict heart disease!')

if submit_button:
    with st.spinner("Predicting your heart disease..."):
        result = heart(heartdisease, 
                    st.session_state.smoking, 
                    st.session_state.alcoholdrinking, 
                    st.session_state.stroke, 
                    st.session_state.diffwalking, 
                    st.session_state.gender, 
                    st.session_state.agecategory, 
                    st.session_state.race, 
                    st.session_state.diabetic, 
                    st.session_state.physicalactivity, 
                    st.session_state.genhealth, 
                    st.session_state.asthma, 
                    st.session_state.kidneydisease, 
                    st.session_state.skincancer, 
                    mentalhealth, 
                    physicalhealth, 
                    sleeptime, 
                    bmi)
        st.success(result)
        st.caption("Made with ❤️ by Shaheer Jamal")