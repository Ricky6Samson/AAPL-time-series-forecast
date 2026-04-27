def load_and_clean_data(path=r"C:\Users\CSC\Documents\New Portfolio\Time series\data\AAPL_data.csv"):
    import pandas as pd
    df = pd.read_csv(path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    df['date']=pd.to_datetime(df['date'])
    df = df.sort_values('date')
    df.set_index('date', inplace=True)
    df = df.drop(columns=['Name'])
    df.isnull().sum()

    return df

