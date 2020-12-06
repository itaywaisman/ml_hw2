import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
from sklearn.compose import make_column_transformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import FunctionTransformer

import globals


def split_data(df):
    X = df.drop(labels=['TestResultsCode'], axis=1)
    y = df[['TestResultsCode']]
    
    train_ratio = 0.7
    val_ratio = 0.15
    test_ratio = 0.15
    
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio, random_state=1)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=(val_ratio/(val_ratio + train_ratio)), random_state=1)
    X_train, X_val, X_test, y_train, y_val, y_test = X_train.reset_index(drop=True), X_val.reset_index(drop=True), X_test.reset_index(drop=True), y_train.reset_index(drop=True), y_val.reset_index(drop=True), y_test.reset_index(drop=True)
    return X_train, X_val, X_test, y_train, y_val, y_test



def fix_data_types(X):
    convert_features_dict = {
        'BloodType': pd.CategoricalDtype(categories=['AB-', 'A+', 'AB+', 'A-', 'B-', 'O-', 'B+', 'O+']),
        'SyndromeClass': pd.CategoricalDtype(categories=range(1, 5))
    }
    return X.astype(convert_features_dict)


def fix_label_type(y):
    convert_label_dict = {
        'TestResultsCode': pd.CategoricalDtype(categories=globals.label_categories)
    }
    y = y.astype(convert_label_dict)
    return y


def handle_sex_type(X):
    return X.replace({'Sex': {'F': -1, 'M': 1}})

def handle_blood_type(X):
    return pd.get_dummies(X, columns=["BloodType"], prefix=["BloodType"])

def handle_date_of_pcr_test(X):
    X['DateOfPCRTest'] = pd.to_datetime(X['DateOfPCRTest'], infer_datetime_format=True)
    X['DateOfPCRTest'] = X['DateOfPCRTest'].values.astype(float)
    X['DateOfPCRTest'].values[X['DateOfPCRTest'].values < 0] = np.nan
    return X


def handle_syndrome_class(X):
    return pd.get_dummies(X, columns=["SyndromeClass"], prefix=["SyndromeClass"])


def handle_location(X):
    long_lat_df = X['CurrentLocation'].str.strip('(Decimal').str.split(', ', expand=True).rename(columns={0:'Lat', 1:'Long'})
    X['CurrentLocation_Lat'] = long_lat_df['Lat'].str.strip("')")
    X['CurrentLocation_Long'] = long_lat_df['Long'].str.strip("Decimal('").str.rstrip("'))")

    convert_dict = {
        'CurrentLocation_Lat': float,
        'CurrentLocation_Long': float,
    }

    X = X.astype(convert_dict)
    return X.drop(labels=['CurrentLocation'], axis=1)


def handle_symptoms(X):
    splitted_df = X['SelfDeclarationOfIllnessForm'].str.split(';', expand=True)
    values = splitted_df.values.flatten()
    unique_values = pd.unique(values).tolist()
    stripped_unique_values = [str(val).strip(' ') for val in unique_values]

    # Split by ; to create a list for each row
    X['SelfDeclarationOfIllnessForm_list'] = X['SelfDeclarationOfIllnessForm'].str.split(';')

    # Replace NAN values with empty list
    isna = X['SelfDeclarationOfIllnessForm_list'].isna()
    X.loc[isna, 'SelfDeclarationOfIllnessForm_list'] = pd.Series([['nan']] * isna.sum()).values

    # strip whitespaces
    X['SelfDeclarationOfIllnessForm_list'] = [[str(val).strip() for val in list(symptom_list)]
                                              for symptom_list in X['SelfDeclarationOfIllnessForm_list'].values]

    # Create columns
    for column_name in stripped_unique_values:
        X[column_name] = X['SelfDeclarationOfIllnessForm_list'].map(lambda l: 1 if column_name in l else 0)

    # Rename no symptoms column
    # Drop irrelevant features
    X = X.rename(columns={'nan': 'No_Symptoms'})\
        .drop(labels=['SelfDeclarationOfIllnessForm','SelfDeclarationOfIllnessForm_list'], axis=1)
    return X


# Transformers
blood_type_transformer = FunctionTransformer(handle_blood_type)
data_types_transformer = FunctionTransformer(fix_data_types)
label_transformer = FunctionTransformer(fix_label_type)
sex_type_transformer = FunctionTransformer(handle_sex_type)
date_of_pcr_test_type_transformer = FunctionTransformer(handle_date_of_pcr_test)
syndrome_class_transformer = FunctionTransformer(handle_syndrome_class)
location_transformer = FunctionTransformer(handle_location)
symptoms_transformer = FunctionTransformer(handle_symptoms)
features_data_types_pipeline = Pipeline([
    ('handle_sex_type', sex_type_transformer),
    ('handle_blood_type', blood_type_transformer),
    ('handle_date_of_pcr_test', date_of_pcr_test_type_transformer),
    ('handle_syndrome_class', syndrome_class_transformer),
    ('handle_location', location_transformer),
    ('handle_symptoms', symptoms_transformer)
])

label_transformer = FunctionTransformer(fix_label_type)