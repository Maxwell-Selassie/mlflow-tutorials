from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow.pyfunc
import pandas as pd

# Define input schema
class InputData(BaseModel):
    age: float
    sex: float
    bmi: float
    bp: float 
    s1: float 
    s2: float
    s3: float 
    s4: float 
    s5: float 
    s6: float 

# initialize app
app = FastAPI(title='Diabetes Prediction API')

# load production model
model_name = 'diabetes_prediction'
model = mlflow.pyfunc.load_model(f'models:/{model_name}/Production')


# prediction endpoint
@app.post('/predict')
def predict(data: InputData):
    try:
        df = pd.DataFrame([data.dict()])
        preds = model.predict(df)
        return {'prediction': preds.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))  