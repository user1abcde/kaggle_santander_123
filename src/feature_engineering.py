def enhance_with_timeseries_features(X, prediction_features):
    X_pred = X[prediction_features]

    how_long_had = X_pred.groupby(X['Customer_Code']).cumsum()
    ever_had = how_long_had > 0
    had_and_does_not_have_now = ever_had & ~(X_pred.astype('bool'))

    for ft in prediction_features:
        X['HAD_NOT_NOW_' + ft.replace('FT_', '')] = had_and_does_not_have_now[ft].astype('int64')
