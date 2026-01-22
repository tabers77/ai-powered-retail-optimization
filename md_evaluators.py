import numpy as np
import pandas as pd
from sklearn import model_selection as ms
import sklearn.metrics as metrics
# Local files
from md_configs import setUp
import md_helpers as helpers


class ModelEvaluators:
    def __init__(self):
        pass


def compute_hit_score(predictions_proba, y_test, top_n=3):
    # Observe that here we assume that predictions_proba has a certain format
    score = 0
    for i in range(len(y_test)):
        if list(y_test)[i] in predictions_proba.transpose()[i].sort_values(ascending=False)[:top_n].index:
            score += 1
    return score / len(y_test)

def compute_custom_mape(y_test, predictions):
    # Replace 0 with a small value
    # Add a small value to actual values
    y_test = np.array([0.1 if i == 0 else i for i in y_test])
    predictions = np.array([0.1 if i == 0 else i for i in predictions])
    # Compute absolute percentage errors
    mape = np.abs((y_test - predictions) / y_test)
    # Compute mean absolute percentage error
    mape = np.mean(mape)
    return mape

def compute_mse_mae_rmse(y_test, predictions):
    mae = metrics.mean_absolute_error(y_test, predictions)
    mse = metrics.mean_squared_error(y_test, predictions)
    rmse = np.sqrt(metrics.mean_squared_error(y_test, predictions))
    mape = compute_custom_mape(y_test, predictions)

    return mae, mse, rmse, mape


def classification_scorers(y_test, predictions):
    acc = metrics.accuracy_score(y_test, predictions)
    f1 = metrics.f1_score(y_test, predictions, average='macro')
    print(metrics.classification_report(y_test, predictions))
    return acc, f1


# TODO: REMOVE setUp['resample_method']
def get_time_series_cv_score(unique_barcode_df, n_splits: int, model, resample_method):
    """Applies cross validation to a time series dataset"""

    x, y = unique_barcode_df.drop(setUp['target_variable'], axis=1), unique_barcode_df[setUp['target_variable']]

    x = x[helpers.generate_selected_features(resample_method)]

    tscv = ms.TimeSeriesSplit(n_splits=n_splits)

    scores = []

    for train_index, test_index in tscv.split(x, y):
        x_train, x_test = x.iloc[train_index], x.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        # print(f'Train shape: {x_train.shape}')
        # print(f'Test shape: {x_test.shape}')
        model.fit(x_train, y_train)

        y_pred = model.predict(x_test)
        score = metrics.mean_squared_error(y_test, y_pred)
        scores.append(score)

    # print(f'Cross validation MSE: {np.mean(scores)}')

    return np.mean(scores)


def custom_mse(actuals, pred, thresh=1.1):
    score = 0
    for i in range(len(actuals)):
        if actuals[i] == 0:
            actuals[i], pred[i] = actuals[i] + 1, pred[i] + 1

        diff = (actuals[i] - pred[i]) ** 2

        pct_comp = diff / actuals[i]

        if pct_comp <= thresh:
            score += 1

    return score / len(actuals)


def aggregate_mse_mae(model,
                      predictions,
                      y_test,
                      resample_method,
                      timeseries_dataframe=None,
                      use_cv_score=False):
    errors_hash = dict()
    mae, mse, rmse, mape = compute_mse_mae_rmse(y_test, predictions)

    errors_hash['mse'] = mse
    errors_hash['mae'] = mae

    if use_cv_score:
        # Cross validation
        cv_score = get_time_series_cv_score(unique_barcode_df=timeseries_dataframe, n_splits=3,
                                            model=model, resample_method=resample_method)
        errors_hash['cv_score'] = cv_score
        return errors_hash

    return errors_hash


# ----------------
# ERROR AGGREGATION LEVELS
# ----------------
def s1_collect_errors_from_error_hash(error_container, errors_hash):
    """
    Example :  {'mse':[1,2,3,4]}
    :param error_container:
    :param errors_hash:
    :return:
    """
    for error_name, value_error in errors_hash.items():
        error_container.setdefault(error_name, []).append(value_error)


def s2_aggregate_individual_errors(model_name, cab_name, comparison, error_container):
    mean_agg = {f'mean_agg_{k}': np.mean(v) for k, v in error_container.items()}
    std_agg = {f'std_agg_{k}': np.std(v) for k, v in error_container.items()}

    comparison.setdefault((model_name, cab_name), []).append(mean_agg)
    comparison.setdefault((model_name, cab_name), []).append(std_agg)
    return comparison, mean_agg, std_agg


def s3_aggregate_all_level_values(comparison: dict):
    container_results = {}
    for key, values in comparison.items():
        algorithm = key[0]
        for row in values:
            for k, v in row.items():
                if k == 'mean_agg_mse':
                    container_results.setdefault(algorithm, ([], [], [], [], []))[0].append(v)
                if k == 'mean_agg_mae':
                    container_results.setdefault(algorithm, ([], [], [], [], []))[1].append(v)
                if k == 'mean_agg_rmse':
                    container_results.setdefault(algorithm, ([], [], [], [], []))[2].append(v)
                if k == 'mean_agg_mape':
                    container_results.setdefault(algorithm, ([], [], [], [], []))[3].append(v)
                if k == 'mean_agg_time_total':
                    container_results.setdefault(algorithm, ([], [], [], [], []))[4].append(v)

    for algorithm, values in container_results.items():
        container_results[algorithm] = (sum(values[0]) / len(values[0]), sum(values[1]) / len(values[1]),
                                        sum(values[2]) / len(values[2]), sum(values[3]) / len(values[3]),
                                        sum(values[4]) / len(values[4]))
    summary_results_table = pd.DataFrame.from_dict(container_results, orient='index').reset_index().rename(columns=
    {
        'index': 'model_name',
        0: 'mean_agg_mse',
        1: 'mean_agg_mae',
        2: 'mean_agg_rmse',
        3: 'mean_agg_mape',
        4: 'mean_agg_time_total',

    })
    return summary_results_table
