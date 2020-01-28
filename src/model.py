from sklearn.multiclass import OneVsRestClassifier
import numpy as np
import xgboost

from src.metrics import transform_y, mapk


def get_xgb_model():
    return OneVsRestClassifier(n_jobs=-1, estimator=xgboost.XGBClassifier(tree_method='hist', max_depth=6))


def fit_model(model, X, Y):
    prediction_features = list(Y.columns)

    xgb_X = _prepare_xgb_data(X)
    xgb_Y = (xgb_X[prediction_features] + Y).fillna(0)

    model.fit(xgb_X, xgb_Y)


def predict_proba(model, X, prediction_features):
    xgb_X = _prepare_xgb_data(X)
    probas = model.predict_proba(xgb_X) * ~xgb_X[prediction_features].astype('bool')

    return probas


def predict_ordered_list(probas):
    return [_predict_row(row, 7, 0.0001) for row in probas.values]


def evaluate_result(pred_ordered_lists, Y):
    return mapk(transform_y(Y, thresh=0.01), pred_ordered_lists, k=7)


def _predict_row(row, n_labels, thresh):
    res = np.argsort(-row)[:n_labels]
    res = res[row[res] >= thresh]
    return res


def _prepare_xgb_data(X):
    xgb_X = X.drop(['Row_Date', 'Customer_Code'], axis=1)

    # xgb_X = _enhance_with_province_data(xgb_X)

    xgb_X['Province_Name'] = xgb_X['Province_Name'].apply(lambda s: 1 if s == 'MADRID' else 0).astype('float64')
    xgb_X['Sex'] = xgb_X['Sex'].apply(lambda s: 1 if s == 'V' else 0).astype('float64')
    xgb_X['Segmentation'] = xgb_X['Segmentation'].apply(
        lambda s: 1 if 'TOP' in s else 0 if 'PARTICULARES' in s else -1).astype('float64')

    return xgb_X


def _enhance_with_province_data(X):
    from src.province import get_province_data

    province_df = get_province_data()

    formated_province = X['Province_Name'].str.replace(",.*$", "")
    X['Province_GDP'] = formated_province.apply(lambda s: province_df.loc[s]['gdp']).astype('float64')
    X['Province_Density'] = formated_province.apply(lambda s: province_df.loc[s]['density']).astype('float64')

    return X
