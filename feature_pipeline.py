import modal

LOCAL=False
N_SAMPLES=4

if LOCAL == False:
   stub = modal.Stub("heart_daily")
   image = modal.Image.debian_slim().pip_install(["hopsworks", "ydata-synthetic==1.3.2", "pandas", "scikit-learn==1.3.2", "joblib", "numpy"])

   @stub.function(image=image, schedule=modal.Period(days=1), secret=modal.Secret.from_name("id2223"))
   def f():
       g()

import joblib
import pandas as pd
import numpy as np

def g():
    import hopsworks
    from datetime import datetime
    import pandas as pd

    project = hopsworks.login()
    fs = project.get_feature_store()

    heart_fg = fs.get_feature_group(name="heart", version=1)

    # Get all user data
    user_fg = fs.get_feature_group(name="heart_user_dataset", version=1)
    df = user_fg.read()
    df["timestamp"] = pd.to_datetime(datetime.now())
    print(df)


if __name__ == "__main__":
    if LOCAL == True:
        g()
    else:
        stub.deploy("heart_daily")
        with stub.run():
            f()