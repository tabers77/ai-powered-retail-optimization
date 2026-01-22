setUp = {'target_variable': 'TotalCount',
         'resample_method': 'H',
         'DEFAULT_SEED': 0,
         'AGG_DATE_COL': 'Date_Hour'}

init_agg_cols = ['Date_Hour', 'barcode', 'Year', 'Month', 'week', 'WeekDay', 'Day', 'Hour']


class FeaturesSpace:
    hour_agg = ['Year', 'Month', 'Day', 'Hour']
    hour_agg_features = ['Year', 'Month', 'week', 'Day', 'WeekDay', 'Hour', 'is_weekend', 'month_sin', 'month_cos',
                         'day_sin', 'day_cos', 'hour_sin', 'hour_cos']
    day_agg = ['Year', 'Month', 'Day']

    day_agg_features = ['Year', 'Month', 'week', 'Day', 'WeekDay', 'is_weekend', 'month_sin', 'month_cos',
                        'day_sin', 'day_cos']

    week_agg = ['Year', 'Month', 'Day']

    week_agg_features = ['Year', 'Month', 'Day', 'week', 'month_sin', 'month_cos']

    weather_features = ['temperature', 'humidity']

    holidays_features = ['is_holiday']


rf_param_grid = {'n_estimators': [i for i in range(200, 2100, 100)],
                 'max_depth': [i for i in range(10, 120, 10)],

                 'max_features': [i for i in range(1, 6, 2)],
                 'min_samples_split': [i / 10.0 for i in range(1, 5)],
                 'min_samples_leaf': [1, 2, 4],
                 'bootstrap': [True, False]
                 }
