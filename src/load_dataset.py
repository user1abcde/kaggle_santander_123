import pandas as pd
import dask.dataframe as dd
import math

BLOCK_SIZE = 10e6
TRANSLATION_DICT_PATH = "static/feature_translation.csv"


def split_train_test(dataset, train_ratio, random_seed):
    customers = pd.Series(dataset.index.unique())
    train_size = math.floor(len(customers) * train_ratio)
    train_customers = set(customers.sample(train_size, random_state=random_seed).values)

    train_df = dataset.loc[dataset.index.isin(train_customers)]
    test_df = dataset.loc[~dataset.index.isin(train_customers)]

    return train_df, test_df


def load_dataset(csv_path, customer_count, random_seed):
    df = dd.read_csv(csv_path, blocksize=BLOCK_SIZE,
                     dtype={'ind_nom_pens_ult1': 'float64', 'ind_nomina_ult1': 'float64',
                            'conyuemp': 'object', 'indrel_1mes': 'object'})

    translation_dict, _ = get_feature_translation_dict(TRANSLATION_DICT_PATH)

    df = df.rename(columns=translation_dict)
    indexed_df = df.set_index('Customer_Code')
    customers = indexed_df.index.unique().compute()

    chosen_customers = set(customers.sample(customer_count, random_state=random_seed).values)
    partial_df = indexed_df.loc[indexed_df.index.isin(chosen_customers)].compute()
    partial_df = _clean_dataframe(partial_df)

    return partial_df


def load_entire_dataset(csv_path, savedir_path, customer_block_size=100000):
    df = dd.read_csv(csv_path, blocksize=BLOCK_SIZE,
                     dtype={'ind_nom_pens_ult1': 'float64', 'ind_nomina_ult1': 'float64',
                            'conyuemp': 'object', 'indrel_1mes': 'object'})

    translation_dict, _ = get_feature_translation_dict(TRANSLATION_DICT_PATH)

    df = df.rename(columns=translation_dict)
    indexed_df = df.set_index('Customer_Code')
    customers = indexed_df.index.unique().compute()

    print("loaded customers set")

    for i in range(math.ceil(len(customers) / customer_block_size)):
        print(f"computing {i} batch")
        chosen_customers = set(customers[i*customer_block_size: (i+1)*customer_block_size])
        partial_df: pd.DataFrame = indexed_df.loc[indexed_df.index.isin(chosen_customers)].compute()
        partial_df = _clean_dataframe(partial_df)
        partial_df.to_pickle(savedir_path + f'/block{i}.pkl')
        print(f"{i} batch complete")


def get_feature_translation_dict(dict_path):
    translation_df = pd.read_csv(dict_path)
    translation_dict = translation_df.set_index('original').to_dict()['translation']
    reverse_translation_dict = translation_df.set_index('translation').to_dict()['original']

    return translation_dict, reverse_translation_dict


def _clean_dataframe(df):
    df['Row_Date'] = pd.to_datetime(df['Row_Date'])
    df['Last_Date_primary'] = pd.to_datetime(df['Last_Date_primary'])
    df['Registration_Date'] = pd.to_datetime(df['Registration_Date'])

    df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
    df['Customer_Seniority_Months'] = pd.to_numeric(df['Customer_Seniority_Months'],
                                                    errors='coerce')

    df['Is_Residence_Different_Bank'] = df['Is_Residence_Different_Bank'].map(
        {'S': 1, 'N': 0, 1: 1, 0: 0}).astype('float64')
    df['Is_Birth_Different_Bank'] = df['Is_Birth_Different_Bank'].map(
        {'S': 1, 'N': 0, 1: 1, 0: 0}).astype('float64')
    df['Is_Spouse_Of_Employee'] = df['Is_Spouse_Of_Employee'].map(
        {'S': 1, 'N': 0, 1: 1, 0: 0}).astype('float64')
    df['Is_Deceased'] = df['Is_Deceased'].map({'S': 1, 'N': 0, 1: 1, 0: 0}).astype('float64')

    categorical_fields = df.dtypes[df.dtypes == 'object'].index

    for field in categorical_fields:
        df[field] = df[field].astype('category')

    return df
