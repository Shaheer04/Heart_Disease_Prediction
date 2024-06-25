import hopsworks
import joblib


# Connect to the Feature Store
project = hopsworks.login(api_key_file='featurestore.key', project='heartdisease')
print("Connected to the Feature Store")

#Get feature store 
fs = project.get_feature_store()
mr = project.get_model_registry()
model = mr.get_model("heart_model_v1", version=1)

model_dir = model.download()

#Load Model and Preprocessing Pipeline
model = joblib.load("../heart_model/heart_model.pkl")
preprocessing_pipeline = joblib.load("../heart_model/preprocessing_pipeline.pkl")
print("Model downloaded")

columns = ['smoking','alcohol_drinking','stroke','diff_walking','sex','age_category','race','diabetic','physical_activity','gen_health','asthma','kidney_disease','skin_cancer','b_m_i','mental_health','physical_health','sleep_time']

# heart_fg = fs.get_feature_group(name="heart", version=1)
heart_fg = fs.get_or_create_feature_group(
    name="heart_user_dataset",
    version=1,
    primary_key = columns,
    description="Heart Dataset of User Input Values",
    event_time="timestamp",
                )