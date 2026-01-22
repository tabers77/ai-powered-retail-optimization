import numpy as np
import pandas as pd
import md_utils as utils
import md_helpers as h
from md_configs import setUp, init_agg_cols
from datetime import datetime, timedelta

from models_tests.tests_utils import df_has_duplicate_rows

class TimeSeriesPreprocessor:
    """
    The TimeSeriesPreprocessor class provides methods to preprocess and manipulate time series data. The class has
    five static methods, each of which represents a preprocessing step, and a run_steps method that combines the
    preprocessing steps to return a time series dataframe for a given barcode in a given cabinet.

    The static methods are:

    >> step1_prepare_for_ts(unique_barcode_df, resample_method='H'): This method prepares the time series data by
    filtering
    the dataframe to contain only the relevant features, checking that the date column is in the correct format,
    resampling the dataframe to the specified frequency, filling missing values, and checking that the target variable
    sums to the same value before and after filling the missing values.

    >> step2_date_range_equalizer(resample_method, unique_barcode_df): This method equalizes the date range of the time
    series data by adding rows with zeros for any missing dates.

    >> step3_get_time_features(unique_barcode_df, date_col=setUp['AGG_DATE_COL']): This method adds time-related features
    to the time series data, such as year, month, day, hour, and cyclical encodings for certain features.

    >> step4_add_weather(self, cabinet_df, timeseries_df): This method adds weather data to the time series data for a
    given cabinet.

    >> step5_column_handler(timeseries_df, resample_method): This method removes certain features from the time series
    data based on the specified resampling method.

    The run_steps method takes in a barcode and a cabinet_df and returns a time series dataframe for the specified
    barcode. The resample_method argument can be used to specify the frequency of the resulting time series data. The
    method combines the above static methods to produce the final time series dataframe.


    """

    def __init__(self, weather_df=None, holidays_df=None):
        # Observe that this dataframe is an aggregation of the original dataset
        # The dataset has to be aggregated in the following way:
        # weather_df.groupby(['Date','NewLocation']).agg({'temperature':'mean', 'humidity': 'mean'}).reset_index()
        self.weather_df = weather_df
        self.holidays_df = holidays_df

    @staticmethod
    def step1_prepare_for_ts(unique_barcode_df, resample_method='H'):

        # GET FEATURES - Here we exclude time features that will be added in step3_get_time_features()
        default_features = [setUp['AGG_DATE_COL'], 'barcode', setUp['target_variable']]
        unique_barcode_df = unique_barcode_df[default_features]

        # Check that Date is in the correct format
        unique_barcode_df[setUp['AGG_DATE_COL']] = pd.to_datetime(unique_barcode_df[setUp['AGG_DATE_COL']])
        # Here we store the barcode in a variable
        barcode = str(unique_barcode_df.barcode.unique()[0])
        unique_barcode_df = unique_barcode_df.set_index(setUp['AGG_DATE_COL']).resample(resample_method).agg(
            {'TotalCount': sum}).reset_index()

        # Fix the barcode format
        unique_barcode_df['barcode'] = barcode
        unique_barcode_df.replace(0, np.nan, inplace=True)

        # DATASET PREP FOR TIME SERIES - FILL MISSING VALUES
        unique_barcode_df['barcode'] = unique_barcode_df['barcode'].fillna(method='ffill')
        unique_barcode_df[setUp['target_variable']] = unique_barcode_df[setUp['target_variable']].fillna(0)

        assert unique_barcode_df[setUp['target_variable']].sum() == unique_barcode_df[setUp['target_variable']].sum()

        return unique_barcode_df

    @staticmethod
    def step2_date_range_equalizer(resample_method, unique_barcode_df):
        # This section adds zeros to obtain a full time range
        unique_barcode_df.set_index(setUp['AGG_DATE_COL'], inplace=True)

        end_date = datetime.now()
        while end_date.weekday() != 6:
            end_date -= timedelta(days=1)

        end_date_str = datetime.strftime(end_date, '%Y-%m-%d')
        new_date_rng = pd.date_range(start=unique_barcode_df.index.min(), end=end_date_str, freq=resample_method)
        unique_barcode_df = unique_barcode_df.reindex(new_date_rng).reset_index().rename(columns={'index': 'Date_Hour'})
        unique_barcode_df['barcode'] = unique_barcode_df['barcode'].fillna(method='ffill')
        unique_barcode_df[setUp['target_variable']] = unique_barcode_df[setUp['target_variable']].fillna(0)
        return unique_barcode_df


    @staticmethod
    def is_date_holiday(date, location, holidays_df):
        holidays = set(holidays_df[holidays_df.NewLocation == location]['month_day'])
        month_day = datetime.strftime(pd.to_datetime(date), '%m-%d')
        if month_day in holidays:
            return 1
        else:
            return 0

    def step3_get_time_features(self, timeseries_df, cabinet_df=None, date_col=setUp['AGG_DATE_COL']):
        timeseries_df['Year'] = timeseries_df[date_col].apply(lambda x: x.year)
        timeseries_df['Month'] = timeseries_df[date_col].apply(lambda x: x.month)
        timeseries_df['WeekDay'] = timeseries_df[date_col].apply(lambda x: x.dayofweek)
        timeseries_df['week'] = timeseries_df[date_col].apply(lambda x: x.week)
        timeseries_df['Day'] = timeseries_df[date_col].apply(lambda x: x.day)
        timeseries_df['Hour'] = timeseries_df[date_col].apply(lambda x: x.hour)
        timeseries_df['is_weekend'] = timeseries_df['WeekDay'].apply(lambda x: 1 if x in (5, 6) else 0)

        # Obtain the location of the cabinet
        if self.holidays_df is not None and cabinet_df is not None:
            location = cabinet_df.NewLocation.unique().item()
            timeseries_df['is_holiday'] = timeseries_df[date_col].apply(
                lambda date: self.is_date_holiday(date, location, self.holidays_df))

        # ------------------ Cyclical one_hot_label_encode ------------------
        # When we include the information about the month of the
        # observed consumption, it makes sense there is a stronger connection between two consecutive months. Using this
        # logic, the connection between December and January and between January and February is strong.
        timeseries_df = utils.apply_cyclical_encoding(timeseries_df)

        return timeseries_df

    def step4_add_weather(self, cabinet_df, timeseries_df):
        # Add keys to time  series
        # Observe that cabinet_df contains all columns
        timeseries_df['DeviceName'] = cabinet_df.DeviceName.unique().item()
        timeseries_df['NewLocation'] = cabinet_df.NewLocation.unique().item()
        self.weather_df.Date = pd.to_datetime(self.weather_df.Date)
        # There are no duplicates
        timeseries_df = pd.merge(timeseries_df.rename(columns={'Date_Hour': 'Date'}), self.weather_df, how='left',
                                 on=['Date', 'NewLocation'])
        timeseries_df.fillna(9999, inplace=True)
        timeseries_df.drop(['DeviceName', 'NewLocation'], axis=1, inplace=True)
        return timeseries_df

    def step5_column_handler(self, timeseries_df, resample_method):
        # Observe that this method applies changes to the dataframe inplace
        # Observe that only if we resample by day the feature related to weather will be added
        if resample_method in ['d', 'D']:
            timeseries_df.drop(['Hour', 'hour_cos', 'hour_sin'], inplace=True, axis=1)

            #   TODO: REFACTOR THIS PART
            if self.weather_df is not None and self.holidays_df is not None:
                features = h.generate_selected_features(resample_method, use_weather=True, use_holidays=True)
            elif self.weather_df is not None and self.holidays_df is None:
                features = h.generate_selected_features(resample_method, use_weather=True, use_holidays=False)
            elif self.weather_df is None and self.holidays_df is not None:
                features = h.generate_selected_features(resample_method, use_weather=False, use_holidays=True)
            else:
                features = h.generate_selected_features(resample_method, use_weather=False, use_holidays=False)

        # TODO: remove weekly option to reduce complexity
        elif resample_method == 'W':

            timeseries_df.drop(['day_cos', 'day_sin', 'WeekDay', 'Hour', 'hour_cos', 'hour_sin', 'is_weekend'],
                               inplace=True, axis=1)
            features = h.generate_selected_features(resample_method, use_weather=False)
        else:
            raise ValueError('Please choose a valid resample method')

        features.append(setUp['target_variable'])
        timeseries_df = timeseries_df[features]

        return timeseries_df

    @staticmethod
    def step6_sanity_check(timeseries_df):
        if df_has_duplicate_rows(timeseries_df):
            raise ValueError('There are duplicates after aggregation')

    def run_steps(self, barcode, cabinet_df, resample_method=setUp['resample_method']):
        """
        Get a time series dataframe for a given barcode in a given cabinet.

        Parameters:
        - barcode (str): The barcode of the item_dataframe to get the time series data for.
        - cab (pandas DataFrame): A DataFrame containing data for the cabinet.
        - resample_method (str, optional): The resampling method to use. Defaults to the value in the setUp dictionary.

        Returns:
        - pandas DataFrame: A time series dataframe for the given barcode.
        """
        # Get the item_dataframe
        item_dataframe = cabinet_df[cabinet_df.barcode.isin([barcode])]

        # Aggregate
        item_agg = item_dataframe.groupby(init_agg_cols)[setUp['target_variable']].sum().reset_index()

        # 1-2. Prepare dataset for time series and apply resample method
        timeseries_df = self.step1_prepare_for_ts(unique_barcode_df=item_agg, resample_method=resample_method)
        timeseries_df = self.step2_date_range_equalizer(resample_method, timeseries_df)
        # 3. Add additional features to the uni-variate dataset
        timeseries_df = self.step3_get_time_features(timeseries_df, cabinet_df=cabinet_df,
                                                     date_col=setUp['AGG_DATE_COL'])

        # 4. Add weather - The merge only support resample_method d, and W and not H
        if self.weather_df is not None:
            if resample_method != 'H':
                # Observe that cabinet_df contains weather data and should contain holidays
                timeseries_df = self.step4_add_weather(cabinet_df, timeseries_df)
            else:
                raise ValueError('Only aggregations d,D, or W are accepted. Aggregation H is not supported')

        # 5. Remove columns that are not relevant for a certain resampling method
        timeseries_df = self.step5_column_handler(timeseries_df, resample_method)
        self.step6_sanity_check(timeseries_df)

        return timeseries_df
