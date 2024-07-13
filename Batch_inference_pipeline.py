HOURS=24

def g():
    import pandas as pd
    import hopsworks
    import joblib
    from datetime import datetime, timedelta
    from dotenv import load_dotenv,find_dotenv
    import os
    from UI.load_model import model, preprocessing_pipeline


    load_dotenv(find_dotenv())
    HOPSWORKS_KEY = os.getenv("API_KEY")
    project = hopsworks.login(api_key_value=HOPSWORKS_KEY, project='heartdisease')
    fs = project.get_feature_store()

    mr = project.get_model_registry()
    model = mr.get_model("heart_model_v1", version=1)
    
    #model = joblib.load("heart_model/heart_model.pkl")
    #preprocessing_pipeline = joblib.load("heart_model/preprocessing_pipeline.pkl")
    
    fg = fs.get_feature_group(name="heart_user_dataset", version=1)
    df = fg.read(read_options={"use_hive": True})

    if df.shape[0] <= 5:
        print("No new data to predict right now!")
        return
    
    # Filter so we get only last data added
    now = datetime.now()
    diff = now - timedelta(hours=HOURS)

    df['clean_timestamp'] = pd.to_datetime(df.timestamp).dt.tz_localize(None)
    df = df[df['clean_timestamp'] >= diff]    

    # remove clean_timestamp 
    df = df.drop(columns=['clean_timestamp'])
    
    # Hacky fix due to Hopsworks Magic
    df["timestamp"] = df['timestamp'] - pd.to_timedelta(0 * df.index, unit='s')

    y_true = df['heart_disease']
    df = df.drop(columns=['heart_disease'])

    try:
        X = preprocessing_pipeline.transform(df)
        y_pred = model.predict(X)
    except:
        print("Error in the model or preprocessing pipeline!")
        return  

    # Store predictions and truth
    print("Storing predictions in Monitor_df")
    print(df)
    monitor_df = pd.DataFrame({"pred": y_pred, "true": y_true, "timestamp": df['timestamp']})
    concat_df = pd.concat([df, monitor_df], axis=1)
    monitor_fg = fs.get_or_create_feature_group(
        name="heart_predictions",
        version=1,
        primary_key=concat_df.columns,
        description="Heart Monitoring Predictions",
        event_time="timestamp",
    )
    monitor_fg.insert(monitor_df, write_options={"wait_for_job": False})

    print("Finished Prediction on the user data!")

g()