from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error



def get_models():
    lr = LinearRegression()
    xgb = XGBRegressor()
    return lr, xgb



def train_models(xtrain, ytrain):
    lr, xgb = get_models()

    lr.fit(xtrain, ytrain)
    xgb.fit(xtrain, ytrain)

    return lr, xgb



def walk_forward_validation(df, features, train_size, test_size):

    lr, xgb = get_models()

    errors_lr = []
    errors_xgb = []

    preds_lr = []
    preds_xgb = []
    actuals = []

    for i in range(train_size, len(df) - test_size, test_size):

        train = df.iloc[:i]
        test = df.iloc[i:i+test_size]

        xtrain, ytrain = train[features], train['target']
        xtest, ytest = test[features], test['target']

        # Linear Regression
        lr.fit(xtrain, ytrain)
        lr_pred = lr.predict(xtest)

        # XGBoost
        xgb.fit(xtrain, ytrain)
        xgb_pred = xgb.predict(xtest)

        # Errors
        errors_lr.append(mean_absolute_error(ytest, lr_pred))
        errors_xgb.append(mean_absolute_error(ytest, xgb_pred))

        # Store predictions
        preds_lr.extend(lr_pred)
        preds_xgb.extend(xgb_pred)
        actuals.extend(ytest)

    return {
        "lr_errors": errors_lr,
        "xgb_errors": errors_xgb,
        "lr_preds": preds_lr,
        "xgb_preds": preds_xgb,
        "actuals": actuals
    }