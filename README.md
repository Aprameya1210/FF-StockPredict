# Stock Price Prediction

This project demonstrates a machine learning approach to predict stock price movements based on an external data source.

## 1. Approach and Assumptions

The primary assumption is that the stock price movement is influenced by the change in an external data source, referred to as 'Data'. While other factors can influence stock prices, they are not considered in this model.

The approach involves the following steps:
- Preprocessing the data to create meaningful features.
- Training a machine learning model to predict the change in stock price.
- Exposing the model through a REST API.
- Containerizing the application for easy deployment.

## 2. Data Preprocessing

The data preprocessing is handled by the `src/preprocessing.py` script. The key steps are:

- **Loading Data**: Two datasets, `Data.csv` and `StockPrice.csv`, are loaded.
- **Feature Engineering**:
    - **Lag Features**: Lagged values of the 'Data' column are created to represent past values.
    - **Change Features**: The change in 'Data' from the previous day is calculated.
    - **Rolling Mean**: A rolling mean of the 'Data' is calculated to capture trends.
- **Target Variable**: The target variable is the change in stock price from the previous day.
- **Merging**: The two datasets are merged based on the 'Date' column.
- **Handling Missing Values**: Rows with missing values are dropped.

## 3. Model Selection and Evaluation

A **Random Forest Regressor** is chosen for this task due to its ability to capture non-linear relationships in the data. The model is trained in `src/train.py`.

- **Features**: The model is trained on the following features:
    - `Data_Lag1`: The 'Data' value from the previous day.
    - `Data_Change_PrevDay`: The change in 'Data' from the day before.
    - `Data_Rolling_Mean`: The rolling mean of the 'Data'.
- **Target**: The model predicts the `Price_Change`.
- **Evaluation**: The model's performance is evaluated using R-squared and Root Mean Squared Error (RMSE). The notebook `notebooks/Stock_analysis.ipynb` contains a detailed analysis and evaluation of the model.

## 4. Key Insights and Conclusions

- A simple linear regression model fails to predict the stock price accurately due to the non-linear nature of the data and the time-series trend.
- By predicting the *change* in price instead of the absolute price, the model can better capture the market dynamics.
- The Random Forest model, combined with feature engineering, provides a significant improvement in prediction accuracy.

## 5. API and Docker

The trained model is served through a FastAPI application (`app/main.py`).

- **Endpoint**: The API has a `/predict` endpoint that accepts a POST request with the required features and returns the predicted price change.
- **Docker**: The project includes a `Dockerfile` to containerize the application. The Docker image builds the application, trains the model, and runs the FastAPI server. This allows for easy and reproducible deployment.
