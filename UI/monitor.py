import hopsworks
import streamlit as st
from PIL import Image
    

project = project = hopsworks.login(api_key_file='featurestore.key', project='heartdisease')
fs = project.get_feature_store()

dataset_api = project.get_dataset_api()

dataset_api.download("Resources/images/confusion_matrix_heart.png", overwrite=True)
dataset_api.download("Resources/images/shap_heart.png", overwrite=True)
dataset_api.download("Resources/images/df_recent_heart.png", overwrite=True)


# Set up the page layout
st.set_page_config(layout="wide")

# Display Feature Importance based on SHAP
st.header("Feature Importance based on SHAP")
shap_image = Image.open("shap_heart.png")
st.image(shap_image, caption="Feature Importance based on SHAP", use_column_width=True)

# Create two columns for the next row
col1, col2 = st.columns(2)

# Display Recent Prediction History in the first column
with col1:
    st.header("Recent Prediction History")
    recent_predictions_image = Image.open("df_recent_heart.png")
    st.image(recent_predictions_image, caption="Recent Prediction History", use_column_width=True)

# Display Confusion Matrix with Historical Prediction Performance in the second column
with col2:
    st.header("Confusion Matrix with Historical Prediction Performance")
    confusion_matrix_image = Image.open("confusion_matrix_heart.png")
    st.image(confusion_matrix_image, caption="Confusion Matrix with Historical Prediction Performance", use_column_width=True)
