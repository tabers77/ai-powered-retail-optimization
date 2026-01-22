import mlflow
import numpy as np


# def mlflow_logger(func):
#     @functools.wraps(func)
#     def run_all_steps_f2(log_params, *args, **kwargs):
#         with mlflow.start_run():
#             func(*args, **kwargs)
#             mlflow_log_params(log_params)
#
#     return run_all_steps_f2


def mlflow_logger(n_pct_cabinets, n_days_limit, windows_size, resample_method,
                  model, model_name, cv_errors, mse_errors):
    # External Parameters
    log_external_params("n_pct_cabinets", n_pct_cabinets)
    log_external_params("n_days_limit", n_days_limit)
    log_external_params("windows_size", windows_size)
    log_external_params("resample_method", resample_method)

    # Model Parameters
    log_model_params(model)

    # Metrics
    mlflow.log_metric("mean_mse_error_aggregated", np.mean(mse_errors))
    mlflow.log_metric("mean_cv_error", np.mean(cv_errors))
    # TODO: ADD AGGREGATED RESULTS
    # mlflow.log_metric("error_aggregation", np.mean(cv_errors))

    # Model Logger
    mlflow.sklearn.log_model(model, model_name)


def log_external_params(param_name, param):
    param_name = f'external_param_{param_name}'
    mlflow.log_param(param_name, param)


def log_model_params(model):
    # Observe that  model should have .get_params available
    for name, param_value in model.get_params().items():
        if param_value is not None:
            mlflow.log_param(name, param_value)
