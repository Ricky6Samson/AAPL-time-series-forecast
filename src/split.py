def train_test_split(df, target='target', train_ratio=0.8):

    features = [
    'lag_1', 'lag_2', 'lag_3', 'lag_5', 'lag_10',
    'ma_5', 'ma_10', 'ma_20',
    'volatility_5', 'volatility_10',
    'high_low_range', 'open_close_diff'
    ]
    
    train_size = int(len(df) * train_ratio)

    train = df[:train_size]
    test = df[train_size:]

    xtrain = train[features]
    ytrain = train[target]

    xtest = test[features]
    ytest = test[target]

    return xtrain, xtest, ytrain, ytest