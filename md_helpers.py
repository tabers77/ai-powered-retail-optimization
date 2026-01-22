from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from forecasting.forecasting_preprocessors import TimeSeriesPreprocessor
from md_configs import setUp, FeaturesSpace, init_agg_cols
from sklearn.preprocessing import MinMaxScaler

def get_best_model(dataset_train):
    print('Obtaining best models...')
    scores_container = dict()
    for d_name in dataset_train['DeviceName'].unique():
        cab = dataset_train[dataset_train['DeviceName'] == d_name]
        barcodes = cab['barcode'].unique()
        scores_container[d_name] = {}  # Initialize an empty dictionary for each device name
        for barcode in barcodes:
            p = cab[cab['barcode'] == barcode]
            best_model = p[p['mape'] == p['mape'].min()]['model_name'].unique().item()
            scores_container[d_name][barcode] = best_model
    return scores_container

class SplitsHandler:
    def __init__(self, resample_method):
        self.resample_method = resample_method

    @staticmethod
    def train_test_split(timeseries_dataframe, train_size=0.80, split_method= 'x_y_split'):
        """
        Split a dataset into training and test_data sets.

        Parameters:
        - unique_barcode_df (pandas DataFrame): The dataset to split.
        - train_size (float, optional): The size of the training set, as a fraction of the total size of the dataset.
        Defaults to 0.80.
        - use_split (bool, optional): Whether to split the dataset into training and test_data sets. Defaults to True.

        Returns: - tuple: If `use_split` is True, a tuple containing the training input data, training target data,
        test_data input data, and test_data target data. If `use_split` is False, a tuple containing the input data
        and target data. The input data and target data are both pandas DataFrames.
        """
        # Select the columns available in timeseries_dataframe (weather and holidays will be added  directly)
        x_features = list(timeseries_dataframe.drop(setUp['target_variable'], axis=1).columns)

        # Assertion to check_recommendations_duplicates that the target_variable has not being duplicated
        assert not setUp['target_variable'] in x_features

        # x_y , train_test , x_y_split (default)
        if split_method == 'x_y_split':
            tran_size = int(len(timeseries_dataframe) * train_size)
            train_data, test_data = timeseries_dataframe[:tran_size], timeseries_dataframe[tran_size:]

            x_train, y_train = train_data[x_features], train_data[setUp['target_variable']]
            x_test, y_test = test_data[x_features], test_data[setUp['target_variable']]

            return x_train, y_train, x_test, y_test

        elif split_method == 'x_y':
            x, y = timeseries_dataframe[x_features], timeseries_dataframe[setUp['target_variable']]
            return x, y

        elif split_method == 'train_test':
            tran_size = int(len(timeseries_dataframe) * train_size)
            train_data, test_data = timeseries_dataframe[:tran_size], timeseries_dataframe[tran_size:]
            return train_data, test_data

    @staticmethod
    def multi_lstm_splits(x, y, sliding_window=7, return_only_test=False):
        """
        The function takes in a time series data and splits it into training and testing sets to be used for a LSTM
        model.

        Parameters: x (np.array): An array of shape (n_records, n_features) representing the time series data. y (
        np.array): An array of shape (n_records, n_targets) representing the target variable of the time series.
        sliding_window (int, optional): The size of the sliding window to be used for splitting the time series data
        into training and testing sets. Defaults to 7. test_mode (bool, optional): If True, the function returns only
        the x data, otherwise returns x and y data. Defaults to False.

        Returns: tuple or np.array: A tuple of x and y arrays of shape (n_records-sliding_window, sliding_window,
        n_features) and (n_records-sliding_window, n_targets), respectively if test_mode is False. If test_mode is
        True, the function returns only x data of shape (n_records-sliding_window, sliding_window, n_features).
        """

        n_records = x.shape[0]

        x_container, y_container = [], []

        for i in range(sliding_window, n_records):
            x_container.append(x[i - sliding_window:i])

            if not return_only_test:
                y_container.append(y[i][0])

        if not return_only_test:
            return np.array(x_container), np.array(y_container)
        else:
            return np.array(x_container)

    @staticmethod
    def uni_lstm_splits(unique_barcode_df, train_size, sliding_window=7, scaled=True):
        unique_barcode_df = unique_barcode_df[setUp['target_variable']]
        n_records = unique_barcode_df.shape[0]
        x_container, y_container = [], []

        for i in range(n_records - sliding_window):
            x_container.append(unique_barcode_df[i:(i + sliding_window)])
            y_container.append(unique_barcode_df[i + sliding_window])

        if not scaled:
            x_data, y_data = np.array(x_container), np.array(y_container)
            scaler = MinMaxScaler()
            scaler.fit(y_data.reshape(-1, 1)).flatten()

            tran_size = int(len(unique_barcode_df) * train_size)
            x_train, y_train = np.array(x_data[0:tran_size]), np.array(y_data[0:tran_size])
            x_test, y_test = np.array(x_data[tran_size:len(x_data)]), np.array(y_data[tran_size:len(x_data)])
            return x_train, y_train, x_test, y_test, scaler
        else:
            x_data, y_data = np.array(x_container), np.array(y_container)
            tran_size = int(len(unique_barcode_df) * train_size)
            x_train, y_train = np.array(x_data[0:tran_size]), np.array(y_data[0:tran_size])
            scaler = MinMaxScaler()
            y_train = scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
            x_flat_train = x_train.reshape(-1, 1)
            x_flat_train = scaler.transform(x_flat_train)
            x_train = x_flat_train.reshape(x_train.shape)
            x_test, y_test = np.array(x_data[tran_size:len(x_data)]), np.array(y_data[tran_size:len(x_data)])
            y_test = scaler.transform(y_test.reshape(-1, 1)).flatten()
            x_flat_test = x_test.reshape(-1, 1)
            x_flat_test = scaler.transform(x_flat_test)
            x_test = x_flat_test.reshape(x_test.shape)
            return x_train, y_train, x_test, y_test, scaler


def force_start_date_to_monday(start_date_shifted):
    """
    Converts the input `start_date_shifted` to the nearest Monday if it is not already a Monday, and returns the updated
    date.

    Args:
        start_date_shifted (datetime): A `datetime` object representing the start date.

    Returns:
        datetime: A `datetime` object representing the updated start date, converted to the nearest Monday if necessary.
    """
    if start_date_shifted.weekday() != 0:  # check_recommendations_duplicates if start_date_shifted is not a Monday
        days_to_subtract = start_date_shifted.weekday()  # calculate the number of days to subtract
        start_date_shifted -= timedelta(days=days_to_subtract)
        return start_date_shifted
    else:
        return start_date_shifted


def generate_future_dataset(x, n_days_predicted=7, resample_method='H', cabinet_df=None, weather_df=None,
                            holidays_df=None):
    """
    Generates a future dataset by adding a specified number of days to the last date in an input dataframe.
    The future dataset is resampled using the specified resample method and has additional time features added to it.

    Parameters:
    x (pandas DataFrame): The input dataframe.
    windows_size (int, optional): The number of days to add to the last date in the input dataframe. Default is 7.
    resample_method (str, optional): The resample method to use. Default is 'H' (hourly).

    Returns:
    pandas DataFrame: The future dataset.

    Notes:
    - setUp and selected_features are global variables that should be defined before calling this function. setUp is
      a dictionary that should contain the key 'AGG_DATE_COL' with a string value representing the name of the
      datetime column in the input dataframe.

      selected_features is a list of strings containing the names of the columns to include in the output dataframe.
    """
    # Make a copy of the input dataframe
    x_copy = x.copy()
    # Initialize the object
    ts_prep = TimeSeriesPreprocessor(weather_df=weather_df, holidays_df=holidays_df)

    # Convert year, month, day, and hour columns to datetime
    x_copy[setUp['AGG_DATE_COL']] = pd.to_datetime(x_copy[generate_selected_features(resample_method,
                                                                                     use_train_columns=False)])

    # Calculate the last date in the input dataframe
    # Observe that start_date should be a sunday
    start_date = x_copy[setUp['AGG_DATE_COL']].max()

    # Test that end date is a Sunday
    if start_date.weekday() != 6:
        raise ValueError(f'The max date should end on a Sunday (6) please refer to: '
                         f'{ts_prep.step2_date_range_equalizer.__name__}')

    start_date_shifted = start_date + timedelta(1)
    start_date_shifted = force_start_date_to_monday(start_date_shifted)

    # Calculate the custom windows size by adding the number of days to the windows_size parameter
    # Observe that here we will fill the values that
    custom_windows_size = n_days_predicted - 1

    # Calculate the end date by adding the custom windows size to the last date in the input dataframe
    end_date = start_date_shifted + timedelta(days=custom_windows_size)  # TEST

    # Generate a date range using the start_date_shifted and end_date
    index = pd.date_range(start_date_shifted, end_date)

    # Create an empty dataframe using the date range as the index
    future_dataset = pd.DataFrame(index=index)

    # Resample the data using the resample_method parameter
    future_dataset = future_dataset.resample(resample_method).sum().reset_index().rename(
        columns={'index': setUp['AGG_DATE_COL']})

    # Add additional features
    # Observe that the object is initialized with weather_df and holidays_df
    # At this step holidays should be added
    future_dataset = ts_prep.step3_get_time_features(future_dataset,
                                                     cabinet_df=cabinet_df,
                                                     date_col=setUp['AGG_DATE_COL'])

    # TODO: REFACTOR
    if weather_df is not None and holidays_df is not None:
        if resample_method != 'H':
            future_dataset = ts_prep.step4_add_weather(cabinet_df, future_dataset)
        else:
            raise ValueError('Only aggregations d,D, or W are accepted. Aggregation H is not supported')
        return future_dataset[generate_selected_features(resample_method, use_weather=True, use_holidays=True)]
    elif weather_df is not None and holidays_df is None:
        if resample_method != 'H':
            future_dataset = ts_prep.step4_add_weather(cabinet_df, future_dataset)
        else:
            raise ValueError('Only aggregations d,D, or W are accepted. Aggregation H is not supported')
        return future_dataset[generate_selected_features(resample_method, use_weather=True, use_holidays=False)]
    elif weather_df is None and holidays_df is not None:
        if resample_method != 'H':
            future_dataset = ts_prep.step4_add_weather(cabinet_df, future_dataset)
        else:
            raise ValueError('Only aggregations d,D, or W are accepted. Aggregation H is not supported')
        return future_dataset[generate_selected_features(resample_method, use_weather=False, use_holidays=True)]

    return future_dataset[generate_selected_features(resample_method)]


def generate_selected_features(resample_method, use_train_columns=True, use_weather=False, use_holidays=False):
    """
    This function generates the selected features based on the resample method passed to it.

    :param use_weather:
    :param use_holidays:
    :param use_train_columns:
    :param resample_method: The resample method used for generating the selected features. This can be one of three
    options: 'd' or 'D' for daily resampling, 'W' for weekly resampling, or any other value for monthly resampling.
    :type resample_method: str :return: The selected features for the given resample method :rtype: list

    Example:
    selected_features = generate_selected_features('W')
    print(selected_features)
    Output: ['feature_1', 'feature_3', 'feature_5']


    """
    # Observe that we need to create a copy or list of the object FeaturesSpace in order to manipulate the object
    # to avoid changing the original object
    if resample_method in ['d', 'D']:
        selected_features = list(FeaturesSpace.day_agg_features) if use_train_columns else list(FeaturesSpace.day_agg)

        if use_weather:
            selected_features.extend(list(FeaturesSpace.weather_features))

        if use_holidays:
            selected_features.extend(list(FeaturesSpace.holidays_features))

    #TODO: remove weekly option to reduce complexity
    elif resample_method == 'W':
        selected_features = list(FeaturesSpace.week_agg_features) if use_train_columns else list(FeaturesSpace.week_agg)
    else:
        selected_features = list(FeaturesSpace.hour_agg_features) if use_train_columns else list(FeaturesSpace.hour_agg)

    return selected_features


#TODO: VALIDATE THIS FILTER
def get_top_cabs(copy_df, n_pct_cabinets=0.20):

    total = copy_df['DeviceName'].nunique()
    n_cabinets = int(n_pct_cabinets * total)

    # Calculates the number of cabinets to select based on the percentage provided using the n_pct_cabinets
    # variable and the total number of unique device names.
    top_cabs = list(copy_df.groupby('DeviceName')['totalSumEuro'].sum().reset_index().
                    sort_values('totalSumEuro', ascending=False).head(n_cabinets)['DeviceName'])
    return top_cabs

def get_top_cabs_based_on_barcodes(copy_df, n_pct_cabinets=0.20, n_days_limit=180):

    #TODO:REMOVE BLOCKET BELOW
    # if orgs_ids_list is not None:
    #     copy_df= copy_df[copy_df.Organization_Id.isin(orgs_ids_list)]

    selected_cabinet_list = list()

    top_cabs = get_top_cabs(copy_df, n_pct_cabinets=n_pct_cabinets)

    for device_name in top_cabs:
        cabinet = copy_df[copy_df.DeviceName == device_name]
        # Select barcodes based on n_days_limit
        legit_barcodes = get_top_barcodes(cabinet, n_days_limit=n_days_limit)
        # If the cabinet contains at least 3 barcodes with minimum n_days_limit we include it in the final list
        if len(legit_barcodes) >= 3:
            selected_cabinet_list.append(device_name)

    return selected_cabinet_list


def preprocessor2_cabinet_level(df, cab_name):
    cab = df[df.DeviceName == cab_name]
    # Observe that column Date_Hour is added here
    cab['Date_Hour'] = pd.to_datetime(cab[['Year', 'Month', 'Day', 'Hour']])

    return cab


def get_top_barcodes(cabinet_df, n_days_limit):
    copy_of_cabinet = cabinet_df.copy()
    copy_of_cabinet['Timestamp'] = pd.to_datetime(copy_of_cabinet['Timestamp'])

    barcode_day_df = copy_of_cabinet.groupby('barcode').agg({'Timestamp': lambda x: (x.max() - x.min()).days})
    barcode_day_df = barcode_day_df.reset_index().rename(columns={'Timestamp': 'n_days_active'})
    barcode_day_limit_df = barcode_day_df[barcode_day_df.n_days_active >= n_days_limit]

    return set(barcode_day_limit_df['barcode'])


def get_top_barcodes_wrapper(cabinet_df, n_days_limit, train_size, sparsity_level=0.80):

    selected_barcodes = list()
    top_barcodes = get_top_barcodes(cabinet_df, n_days_limit)
    for barcode in top_barcodes:
        # 1. Get the item_dataframe
        item_dataframe = cabinet_df[cabinet_df.barcode.isin([barcode])]

        # 2. aggregate
        # Initial columns to perform aggregation
        item_agg = item_dataframe.groupby(init_agg_cols)[setUp['target_variable']].sum().reset_index()

        # 3. prepare dataset for time series and apply resample method
        preprocessor = TimeSeriesPreprocessor()
        timeseries_df = preprocessor.step1_prepare_for_ts(item_agg, resample_method='d')
        timeseries_df = preprocessor.step2_date_range_equalizer(resample_method='d', unique_barcode_df=timeseries_df)

        # Here the column count is created after re setting the index. Here we assume that the first character is
        # the count of zero. Observe that using count is valid in pandas 2.0.1
        splitter = SplitsHandler(resample_method= 'd')
        train, test  = splitter.train_test_split(timeseries_df, train_size=train_size, split_method= 'train_test')
        zero_pct_train = (train.TotalCount.value_counts() / len(train)).reset_index().loc[0]['TotalCount']
        zero_pct_test = (test.TotalCount.value_counts() / len(test)).reset_index().loc[0]['TotalCount']
        if zero_pct_train <= sparsity_level and zero_pct_test <= sparsity_level + 0.03:
            selected_barcodes.append(barcode)

    print(f'Percentage of total barcodes after filtering: {len(selected_barcodes) / cabinet_df.barcode.nunique()}')

    return selected_barcodes
