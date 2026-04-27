def create_features(df):
    
    df = df.sort_index()

    df['target'] = df['close'].pct_change().shift(-1)
    df['returns'] = df['close'].pct_change()

    df['lag_1'] = df['close'].shift(1)
    df['lag_2'] = df['close'].shift(2)
    df['lag_3'] = df['close'].shift(3)
    df['lag_5'] = df['close'].shift(5)
    df['lag_10'] = df['close'].shift(10)

    df['ma_5'] = df['close'].rolling(5).mean()
    df['ma_10'] = df['close'].rolling(10).mean()
    df['ma_20'] = df['close'].rolling(20).mean()

    df['volatility_5'] = df['returns'].rolling(5).std()
    df['volatility_10'] = df['returns'].rolling(10).std()

    df['high_low_range'] = df['high'] - df['low']
    df['open_close_diff'] = df['open'] - df['close']

    df.dropna(inplace=True)

    return df