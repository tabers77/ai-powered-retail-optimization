from datetime import datetime
import pandas as pd
import md_helpers as helpers


class ForecastingDatasetFactory:
    @staticmethod
    def build_test_dataset(predictions, x_test, y_test, resample_method):
        x_test, y_test = x_test.reset_index(drop=True), y_test.reset_index(drop=True)
        # Observe that here we deal with many type of outputs depending on the model. For RandomForest we need to
        # convert from list to Series
        if isinstance(predictions, list):
            predictions = pd.Series(predictions)

        predictions = predictions.reset_index(drop=True)
        x_test['actuals'] = y_test
        x_test['predictions'] = predictions
        x_test['datetime'] = pd.to_datetime(x_test[helpers.generate_selected_features(resample_method,
                                                                                      use_train_columns=False)])
        # We use datetime as index make it easier for visualization
        x_test.set_index('datetime', inplace=True)
        return x_test

    @staticmethod
    def build_inferring_dataset(predictions_dataset, x, y, resample_method):
        predictions_dataset['is_predicted'] = 'TRUE'

        full_x = x.copy()
        full_x['actuals'] = y
        # Observe that nulls will be generated at this point
        full_dataset = pd.concat([full_x, predictions_dataset])
        full_dataset['datetime'] = pd.to_datetime(full_dataset[helpers.generate_selected_features(resample_method,
                                                                                                  use_train_columns=False)])
        # We use datetime as index make it easier for visualization
        full_dataset.set_index('datetime', inplace=True)
        return full_dataset

#TODO: REFACTOR
class FeaturesDatasetGenerator:
    @staticmethod
    def generate_weather_df(data_path='data/weather_data_forecasted.csv'):
        """
        Load weather data from CSV file.
        
        Args:
            data_path (str): Path to weather data CSV file
            
        Returns:
            pd.DataFrame: Aggregated weather data by date and location
        """
        weather_df = pd.read_csv(data_path, index_col=0)

        weather_df = weather_df.reset_index().rename(
            columns={'index': 'Date', 'country': 'NewLocation', 'city': 'City'})

        weather_df.drop(['countryName', 'location'], axis=1, inplace=True)
        weather_df.Date = pd.to_datetime(weather_df.Date, utc=True)
        weather_df.Date = weather_df.Date.apply(lambda date: datetime.strftime(date, '%Y-%m-%d'))
        agg_weather = weather_df.groupby(['Date', 'NewLocation']).agg(
            {'temperature': 'mean', 'humidity': 'mean'}).reset_index()
        return agg_weather

    @staticmethod
    def generate_holidays_df(data_path='data/holiday_data.csv'):
        """
        Load holiday data from CSV file.
        
        Args:
            data_path (str): Path to holiday data CSV file
            
        Returns:
            pd.DataFrame: Holiday data with country and month-day information
        """
        holidays_df = pd.read_csv(data_path, index_col=0).reset_index(drop=True)

        # Observe that the format is currently mixed for pandas 2.0.1
        holidays_df['month_day'] = pd.to_datetime(holidays_df.date).apply(lambda date: datetime.strftime(date, '%m-%d'))
        holidays_df = holidays_df[['Country', 'month_day']]
        holidays_df = holidays_df.rename(columns={'Country': 'NewLocation'})
        return holidays_df


