import os
import pandas as pd
from globals import bad_features, continous_features, pre_final_list, pcrs, others, final_features, selected_features
from transformations import split_data, features_data_types_pipeline, label_transformer
from preprocess import Imputer, OutlierClipper, Normalizer
from feature_selection import select_features_filter, select_features_wrapper
from visualize import display_correlation_matrix, save_scatter_plots, plot_df_scatter
from sklearn.pipeline import Pipeline

def save_data_to_csv(X_train, X_val, X_test, y_train, y_val, y_test, suffix='before'):
    train_dataset = pd.concat([X_train, y_train], axis=1)
    val_dataset = pd.concat([X_val, y_val], axis=1)
    test_dataset = pd.concat([X_test, y_test], axis=1)
    
    if not os.path.exists('results'):
        os.makedirs('results')
    
    train_dataset.to_csv(f'results/train_{suffix}.csv')
    val_dataset.to_csv(f'results/val_{suffix}.csv')
    test_dataset.to_csv(f'results/test_{suffix}.csv')

def load_data():
    # Load Dataset
    return pd.read_csv('virus_hw2.csv')
    
def split(df):
    df = df.drop(labels=bad_features, axis=1)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)
    
    return X_train, X_val, X_test, y_train, y_val, y_test
    
def prepare_data(X_train, X_val, X_test, y_train, y_val, y_test):
    
    patient_ids = X_train['PatientID'], X_val['PatientID'], X_test['PatientID']
    X_train, X_val, X_test = X_train.drop(labels=['PatientID'], axis=1), X_val.drop(labels=['PatientID'], axis=1), X_test.drop(labels=['PatientID'], axis=1)
    # Prepare Dataset
    data_preperation_pipelines = Pipeline([
        ('feature_types', features_data_types_pipeline),
        ('feature_imputation', Imputer()),
        ('outlier_clipping', OutlierClipper(features=continous_features)),
        ('normalization', Normalizer())
    ])
    data_preperation_pipelines.fit(X_train, y_train)
    label_transformer.fit(y_train)
    X_train_prepared, y_train_prepared = data_preperation_pipelines.transform(X_train), label_transformer.transform(
        y_train)
    X_validation_prepared, y_validation_prepared = data_preperation_pipelines.transform(
        X_val), label_transformer.transform(y_val)
    X_test_prepared, y_test_prepared = data_preperation_pipelines.transform(X_test), label_transformer.transform(y_test)

    X_train_prepared, X_validation_prepared, X_test_prepared = pd.concat([patient_ids[0], X_train_prepared], axis=1), pd.concat([patient_ids[1], X_validation_prepared], axis=1), pd.concat([patient_ids[2], X_test_prepared], axis=1)

    
    return X_train_prepared, X_validation_prepared, X_test_prepared,\
           y_train_prepared, y_validation_prepared, y_test_prepared

def select_features(X_train_prepared, y_train_prepared):
    sff = select_features_filter(X_train_prepared, y_train_prepared)
    with open('filter_features.txt', 'w') as f:
        f.write(',\n'.join(X_train_prepared.columns[sff.support_]))
    sfs = select_features_wrapper(X_train_prepared, y_train_prepared)
    with open('wrapper_features.txt', 'w') as f:
        f.write(',\n'.join(sfs.k_feature_names_))

    sfs = select_features_wrapper(X_train_prepared[pre_final_list], y_train_prepared, forward=False, k_features=18)


def print_graphs(X_train_prepared, y_train_prepared):
    display_correlation_matrix(X_train_prepared, y_train_prepared)
    display_correlation_matrix(X_train_prepared[list(pre_final_list)], y_train_prepared)
    plot_df_scatter(X_train_prepared[pcrs], 15)
    plot_df_scatter(X_train_prepared[others], 15)
    save_scatter_plots()

def save_selected_features(original_list, final_list):
    df = pd.DataFrame([1 if feature in final_list else 0 for feature in original_list], index=original_list)
    
    df.T.to_csv('selected_columns.csv')
    
if __name__ == '__main__':
    df = load_data()

    X_train, X_val, X_test, y_train, y_val, y_test = split(df)
    save_data_to_csv(X_train, X_val, X_test, y_train, y_val, y_test)
    
    X_train_prepared, X_validation_prepared, X_test_prepared,\
    y_train_prepared, y_validation_prepared, y_test_prepared = prepare_data(X_train, X_val, X_test, y_train, y_val, y_test)
    
#     select_features(X_train_prepared, y_train_prepared)
#     print_graphs(X_train_prepared, y_train_prepared)
    
    X_train_final, X_validation_final, X_test_final = X_train_prepared[final_features], X_validation_prepared[final_features], X_test_prepared[final_features]

    save_data_to_csv(X_train_final, X_validation_final, X_test_final, y_train, y_val, y_test, suffix='after')
    
    save_selected_features(df.columns.drop(labels=['PatientID', 'TestResultsCode']), selected_features)
