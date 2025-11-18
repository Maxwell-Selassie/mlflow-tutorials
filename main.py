'''Production-grade FastAPI service for diabetes prediction'''

import logging
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, Any
import mlflow.pyfunc
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils import setup_logger

# Setup logging
logger = setup_logger('api', 'logs/')

# Constants
MODEL_NAME = 'Diabetes_Prediction_Model'
API_VERSION = '1.0.0'

# Input schema with validation
class InputData(BaseModel):
    age: float = Field(..., ge=-5, le=5, description='Standardized age')
    sex: float = Field(..., ge=-5, le=5, description='Standardized sex')
    bmi: float = Field(..., ge=-5, le=5, description='Standardized BMI')
    bp: float = Field(..., ge=-5, le=5, description='Standardized blood pressure')
    s1: float = Field(..., ge=-5, le=5, description='Standardized serum measurement 1')
    s2: float = Field(..., ge=-5, le=5, description='Standardized serum measurement 2')
    s3: float = Field(..., ge=-5, le=5, description='Standardized serum measurement 3')
    s4: float = Field(..., ge=-5, le=5, description='Standardized serum measurement 4')
    s5: float = Field(..., ge=-5, le=5, description='Standardized serum measurement 5')
    s6: float = Field(..., ge=-5, le=5, description='Standardized serum measurement 6')
    
    class Config:
        json_schema_extra = {
            "example": {
                "age": 0.038,
                "sex": 0.050,
                "bmi": 0.061,
                "bp": 0.021,
                "s1": -0.044,
                "s2": -0.034,
                "s3": -0.043,
                "s4": -0.002,
                "s5": 0.019,
                "s6": -0.017
            }
        }

# Response schema
class PredictionResponse(BaseModel):
    prediction: float
    model_name: str
    model_version: str
    status: str

# Initialize FastAPI app
app = FastAPI(
    title='Diabetes Prediction API',
    description='Production ML API for diabetes disease progression prediction',
    version=API_VERSION
)

# # Add CORS middleware
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # Configure for production
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# Global model variable
model = None

@app.on_event("startup")
async def load_model():
    '''Load production model on startup'''
    global model
    try:
        logger.info(f'Loading model: {MODEL_NAME}/Production')
        model = mlflow.pyfunc.load_model(f'models:/{MODEL_NAME}/Production')
        logger.info('✅ Model loaded successfully')
    except Exception as e:
        logger.error(f'Failed to load model: {e}', exc_info=True)
        raise RuntimeError(f'Model loading failed: {e}')

@app.on_event("shutdown")
async def shutdown():
    '''Cleanup on shutdown'''
    logger.info('API shutting down')

@app.get('/')
def root() -> Dict[str, Any]:
    '''API root endpoint'''
    return {
        'message': 'Diabetes Prediction API',
        'version': API_VERSION,
        'status': 'running',
        'endpoints': {
            'health': '/health',
            'predict': '/predict',
            'model_info': '/model/info',
            'docs': '/docs',
            'openapi': '/openapi.json'
        }
    }

@app.get('/health')
def health_check() -> Dict[str, str]:
    '''Health check endpoint'''
    if model is None:
        logger.error('Health check failed: Model not loaded')
        raise HTTPException(status_code=503, detail='Model not loaded')
    
    return {
        'status': 'healthy',
        'model': MODEL_NAME,
        'version': API_VERSION
    }

@app.get('/model/info')
def model_info() -> Dict[str, Any]:
    '''Return model metadata and expected features'''
    if model is None:
        raise HTTPException(status_code=503, detail='Model not loaded')
    
    return {
        'model_name': MODEL_NAME,
        'stage': 'Production',
        'features': list(InputData.model_fields.keys()),
        'feature_count': len(InputData.model_fields),
        'input_schema': InputData.model_json_schema(),
        'api_version': API_VERSION
    }

@app.post('/predict', response_model=PredictionResponse)
def predict(request: Request, data: InputData) -> PredictionResponse:
    '''
    Make prediction using production model
    
    Args:
        data: Input features (standardized)
        
    Returns:
        Prediction result
    '''
    if model is None:
        logger.error('Prediction failed: Model not loaded')
        raise HTTPException(status_code=503, detail='Model not loaded')
    
    try:
        # Log request
        client_host = request.client.host if request.client else 'unknown'
        logger.info(f'Prediction request from {client_host}')
        
        # Convert to DataFrame
        df = pd.DataFrame([data.dict()])
        
        # Make prediction
        prediction = model.predict(df)
        result = float(prediction[0])
        
        logger.info(f'Prediction successful: {result:.2f}')
        
        return PredictionResponse(
            prediction=result,
            model_name=MODEL_NAME,
            model_version='Production',
            status='success'
        )
        
    except ValueError as e:
        logger.error(f'Validation error: {e}', exc_info=True)
        raise HTTPException(status_code=422, detail=f'Invalid input: {str(e)}')
    
    except Exception as e:
        logger.error(f'Prediction failed: {e}', exc_info=True)
        raise HTTPException(status_code=500, detail=f'Prediction error: {str(e)}')

# Exception handlers
@app.exception_handler(404)
async def not_found_handler(request: Request, exc):
    return {
        'error': 'Not Found',
        'message': f'Endpoint {request.url.path} does not exist',
        'available_endpoints': ['/health', '/predict', '/model/info', '/docs']
    }

# if __name__ == '__main__':
    # import uvicorn
    # uvicorn.run(app, host='0.0.0.0', port=8000)