from fastapi import FastAPI,HTTPException
from contextlib import asynccontextmanager
import joblib
import os
from .schemas import PredictionRequest,PredictionResponse
from src import config

ml_models={}

@asynccontextmanager
async def lifespan(app:FastAPI):
    model_path=config.MODEL_FILE
    
    if not os.path.exists(model_path):
        print(f"Model file not found at {model_path}")
    else:
        ml_models["rf_model"]=joblib.load(model_path)
        print("Model loaded successfully")
    
    yield
    
    ml_models.clear()

app=FastAPI(
    title="Stock Prediction API",
    description="Predicts stock price movements based on 'Data' changes.",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/")
async def root():
    return {"status":"online","message":"Stock Prediction API is running"}

@app.post("/predict",response_model=PredictionResponse)
async def predict_stock_movement(request: PredictionRequest):
    if "rf_model" not in ml_models:
        raise HTTPException(status_code=503,detail="Model is not loaded.")
    
    model=ml_models["rf_model"]

    input_features=[[
        request.data_lag1,
        request.data_change_prev_day,
        request.data_rolling_mean
    ]]
    prediction=model.predict(input_features)[0]

    return {
        "predicted_price_change":round(prediction,4),
        "message": "Prediction successful"
    }

if __name__=="__main__":
    import uvicorn
    uvicorn.run("app.main:app",host="0.0.0.0",port=8000,reload=True)