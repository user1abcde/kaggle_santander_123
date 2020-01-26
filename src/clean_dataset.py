import numpy as np

CONTEXT_FEATURES_TO_KEEP = ['Row_Date', 'Sex', 'Age', 'Customer_Seniority_Months', 'Province_Name', 'Is_Active',
                            'Gross_Household_Income', 'Segmentation']


def clean_dataset(dataset_df, fill_na_context_features=True):
    prediction_features = get_prediction_features(dataset_df)

    all_interesting_features = CONTEXT_FEATURES_TO_KEEP + prediction_features
    clean_dataset = dataset_df[all_interesting_features]

    clean_dataset.loc[clean_dataset['Customer_Seniority_Months'] < 0, 'Customer_Seniority_Months'] = np.nan

    if fill_na_context_features:
        for ft in prediction_features:
            clean_dataset[ft] = clean_dataset[ft].fillna(0)

        clean_dataset['Province_Name'] = clean_dataset['Province_Name'].fillna('MADRID')
        clean_dataset['Segmentation'] = clean_dataset['Segmentation'].fillna('02 - PARTICULARES')
        clean_dataset['Sex'] = clean_dataset['Sex'].fillna('V')

        numeric_dtypes = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
        numeric_columns = clean_dataset.select_dtypes(include=numeric_dtypes).columns
        for ft in numeric_columns:
            clean_dataset[ft] = clean_dataset[ft].fillna(clean_dataset[ft].mean())

    X, Y = split_X_Y(clean_dataset)

    return X, Y


def split_X_Y(clean_dataset_df):
    prediction_features = get_prediction_features(clean_dataset_df)

    gb = clean_dataset_df.groupby(clean_dataset_df.index)

    clean_dataset_df['min'] = gb['Row_Date'].min()
    clean_dataset_df['max'] = gb['Row_Date'].max()

    input_dataset = clean_dataset_df[clean_dataset_df['Row_Date'] != clean_dataset_df['max']].drop(['min', 'max'],
                                                                                                   axis=1)
    output_dataset = clean_dataset_df[clean_dataset_df['Row_Date'] != clean_dataset_df['min']].drop(['min', 'max'],
                                                                                                    axis=1)

    input_dataset = input_dataset.reset_index().sort_values(['Customer_Code', 'Row_Date']).reset_index(drop=True)
    output_dataset = output_dataset.reset_index().sort_values(['Customer_Code', 'Row_Date']).reset_index(drop=True)

    X = input_dataset
    Y = (output_dataset[prediction_features] - input_dataset[prediction_features]).clip(0, 1)

    return X, Y


def reduce_train(train_X, train_Y):
    is_changed_series = train_Y.sum(axis=1) > 0

    X_reduced = train_X[is_changed_series].reset_index(drop=True)
    Y_reduced = train_Y[is_changed_series].reset_index(drop=True)

    return X_reduced, Y_reduced


def get_prediction_features(dataset_df):
    return [col for col in dataset_df.columns if 'FT_' in col]
