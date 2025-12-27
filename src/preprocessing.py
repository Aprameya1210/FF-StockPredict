import pandas as pd
from . import config

def load_and_preprocess():
    # Load Data
    data_df=pd.read_csv(config.DATA_PATH)
    stock_df=pd.read_csv(config.STOCK_PATH)

    # Convert dates from string (python interprets it as string by default) to date format
    data_df['Date']=pd.to_datetime(data_df['Date'])
    stock_df['Date']=pd.to_datetime(stock_df['Date'])

    # Sort the dates
    data_df=data_df.sort_values('Date')
    stock_df=stock_df.sort_values('Date')

    # Calculate lags, changes and rolling means of the full dataset before merging.
    data_df['Data_Lag1']=data_df['Data'].shift(1)
    data_df['Data_Lag2']=data_df['Data'].shift(2)
    data_df['Data_Change_PrevDay']=data_df['Data_Lag1']-data_df['Data_Lag2']
    data_df['Data_Rolling_Mean']=data_df['Data_Lag1'].rolling(window=3).mean()

    # Merge the tables, this safely drops the which have no previous data, but the history is preserved in columns
    df = pd.merge(data_df,stock_df,on='Date',how='inner')

    # Calculate target which indicates the movement of next day price
    df['Price_Change']=df['Price']-df['Price'].shift(1)

    # Removing the first few days as they have no previous data to compare to
    return df.dropna().copy()