from globals import bad_features, continous_features, pre_final_list, pcrs, others
from transformations import split_data, features_data_types_pipeline
from preprocess import Imputer, OutlierClipper, Normalizer
from feature_selection import select_features_filter, select_features_wrapper
from visualize import display_correlation_matrix, save_scatter_plots, plot_df_scatter
from sklearn.pipeline import Pipeline


def load_and_prepare_data():
    # Load Dataset
    df = pd.read_csv('virus_hw2.csv')
    df.drop(labels=bad_features, axis=1, inplace=True)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)

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

    return X_train_prepared, X_validation_prepared, X_test_prepared,\
           y_train_prepared, y_validation_prepared, y_test_prepared


def select_features(X_train_prepared, y_train_prepared):
    sff = select_features_filter(X_train_prepared, y_train_prepared)
    with open('filter_features.txt', 'w') as f:
        f.write(',\n'.join(X_train_prepared.columns[sff.support_]))
    sfs = select_features_wrapper(X_train_prepared, y_train_prepared)
    with open('wrapper_features.txt', 'w') as f:
        f.write(',\n'.join(sfs.k_feature_names_))

    sfs = select_features_wrapper(X_train_prepared[pre_final_list], y_train_prepared, forward=False, k_features=15)


def print_graphs(X_train_prepared, y_train_prepared):
    display_correlation_matrix(X_train_prepared, y_train_prepared)
    display_correlation_matrix(X_train_prepared[list(pre_final_list)], y_train_prepared)
    plot_df_scatter(X_train_prepared[pcrs], 15)
    plot_df_scatter(X_train_prepared[others], 15)
    save_scatter_plots()


if __name__ == '__main__':
    X_train_prepared, X_validation_prepared, X_test_prepared,\
    y_train_prepared, y_validation_prepared, y_test_prepared = load_and_prepare_data()
    #select_features(X_train_prepared, y_train_prepared)
    #print_graphs(X_train_prepared, y_train_prepared)