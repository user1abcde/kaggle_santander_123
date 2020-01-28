def enhance_with_timeseries_features(X, prediction_features):
    """
    :param X: Strong Assumption - X is sorted by [Customer_Code, Row_Date]
    """
    had_and_does_not_have_now = compute_had_and_does_not_have_now(X, prediction_features)
    lagged1 = compute_lagged(X, prediction_features, 1)
    lagged2 = compute_lagged(X, prediction_features, 2)
    lagged3 = compute_lagged(X, prediction_features, 3)
    lagged5 = compute_lagged(X, prediction_features, 5)

    for ft in prediction_features:
        X['HAD_NOT_NOW_' + ft.replace('FT_', '')] = had_and_does_not_have_now[ft].astype('int32')
        X['LAG1_' + ft.replace('FT_', '')] = lagged1[ft].astype('int32')
        X['LAG2_' + ft.replace('FT_', '')] = lagged2[ft].astype('int32')
        X['LAG3_' + ft.replace('FT_', '')] = lagged3[ft].astype('int32')
        X['LAG5_' + ft.replace('FT_', '')] = lagged5[ft].astype('int32')


def compute_had_and_does_not_have_now(X, prediction_features):
    X_pred = X[prediction_features]

    how_long_had = X_pred.groupby(X['Customer_Code']).cumsum()
    ever_had = how_long_had > 0
    had_and_does_not_have_now = ever_had & ~(X_pred.astype('bool'))

    return had_and_does_not_have_now


def compute_lagged(X, prediction_features, lag):
    res = X.groupby('Customer_Code').shift(lag)
    res['Customer_Code'] = X['Customer_Code']
    res = res[['Customer_Code'] + prediction_features].fillna(0)

    return res
