import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from . import config, preprocessing

def train_model():
    # Get clean data
    df = preprocessing.load_and_preprocess()
    print(f"Data loaded, samples: {len(df)}")
    
    # Calculate features such as lags, rolling means etc.
    features = ['Data_Lag1', 'Data_Change_PrevDay', 'Data_Rolling_Mean']
    target = 'Price_Change'

    X = df[features]
    y = df[target]

    # Train model
    print("Training the random forest model")
    model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X, y)

    # Save model
    joblib.dump(model, config.MODEL_FILE)
    print(f"Model saved to {config.MODEL_FILE}...")

if __name__ == "__main__":
    train_model()