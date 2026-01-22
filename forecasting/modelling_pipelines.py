import logging
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import md_evaluators as eval_funcs
from forecasting import forecasting_preprocessors as prep
from md_configs import setUp
from md_dataset_factory import ForecastingDatasetFactory
from models_tests.tests_utils import df_has_duplicate_rows
logging.basicConfig(level=logging.INFO)
logging.getLogger("py4j.java_gateway").setLevel(logging.WARNING)


class ForecastingAPipe:
    # TODO: ADD DOCSTRING
    def __init__(self, resample_method, train_size, infer_mode=False, barcode=None, cabinet_df=None,
                  weather_df=None, holidays_df=None):

        self.y = None
        self.x = None
        self.y_train = None
        self.x_train = None
        self.y_test = None
        self.x_test = None
        self.predictions = None
        self.timeseries_dataframe = None
        self.weather_df = weather_df
        self.holidays_df = holidays_df
        self.barcode = barcode
        self.cabinet_df = cabinet_df
        self.resample_method = resample_method
        self.train_size = train_size
        self.infer_mode = infer_mode
        self.random_state = setUp['DEFAULT_SEED']

    # GLOBAL FUNCTION
    # FEATURES - Observe that this step contains the features so in order to add features modify this step
    def get_preprocessed_ts_df(self):
        ts_prep = prep.TimeSeriesPreprocessor(weather_df=self.weather_df, holidays_df=self.holidays_df)

        self.timeseries_dataframe = ts_prep.run_steps(self.barcode,
                                                      self.cabinet_df,
                                                      resample_method=self.resample_method)



    def train_test_split(self):
        pass

    def train_predict(self):
        pass

    # FEATURES - Observe that this step contains the features so in order to add features modify this step
    def infer_predict(self):
        pass

    def evaluate(self):
        pass

    # GLOBAL
    def build_output(self):
        dataset_builder: ForecastingDatasetFactory = ForecastingDatasetFactory()
        if not self.infer_mode:
            output_dataset = dataset_builder.build_test_dataset(self.predictions,
                                                                self.x_test,
                                                                self.y_test,
                                                                self.resample_method)
        else:
            output_dataset = dataset_builder.build_inferring_dataset(self.predictions,
                                                                     self.x,
                                                                     self.y,
                                                                     self.resample_method)
        if df_has_duplicate_rows(output_dataset):
            raise ValueError('There are duplicates after aggregation, when building output')

        return output_dataset

    @staticmethod
    def weekly_aggregated_evaluation(train_output_dataset):
        errors_hash = dict()
        df = train_output_dataset.groupby(['Year', 'week']).agg({'actuals': sum, 'predictions': sum}).reset_index()

        mse, mae, rmse, mape = eval_funcs.compute_mse_mae_rmse(df.actuals, df.predictions)

        errors_hash['mse'] = mse
        errors_hash['mae'] = mae
        errors_hash['rmse'] = rmse
        errors_hash['mape'] = mape

        return errors_hash

    def run_train_mode(self):
        # 1. Apply preprocessing
        self.get_preprocessed_ts_df()

        # 2. Do the splits
        self.train_test_split()

        # 3. Train and predict
        self.predictions = self.train_predict()

        # 4. Build the output dataset
        output_dataset = self.build_output()

        errors_hash = self.weekly_aggregated_evaluation(output_dataset)

        return output_dataset, errors_hash

    def run_infer_mode(self):
        # 1. Apply preprocessing
        self.get_preprocessed_ts_df()

        # 2. Do the splits
        self.train_test_split()

        # 3. Train and predict
        self.predictions = self.infer_predict()

        # 4. Build the output dataset
        output_dataset = self.build_output()

        return output_dataset

    def run(self):
        if self.infer_mode:
            #print('Running inference mode...')
            output_dataset = self.run_infer_mode()
            return output_dataset
        else:
            #print('Running training mode...')
            output_dataset, errors_hash = self.run_train_mode()
            return output_dataset, errors_hash


class ForecastingBPipe:

    def __init__(self, cabinet_df, barcode, train_size, model, model_name, cab_name):
        self.inverse_mapping = None
        self.predictions_proba = None
        self.predictions = None
        self.y_test = None
        self.x_test = None
        self.y_train = None
        self.x_train = None
        self.product = None
        self.cabinet_df = cabinet_df
        self.barcode = barcode
        self.train_size = train_size
        self.model = model
        self.model_name = model_name
        self.cab_name = cab_name

    def preprocessing(self):
        self.product = self.cabinet_df[self.cabinet_df.barcode == self.barcode]

    def train_test_split(self):
        train_end_index = int(self.train_size * len(self.product['Timestamp'].sort_values()))
        train_end_index = self.product[:train_end_index][-1:].index.tolist()[0]

        train, test = self.cabinet_df.loc[:train_end_index], self.cabinet_df.loc[train_end_index:]

        test = test[test.barcode == self.barcode]

        features = ['CustomerId', 'paidAmount', 'remainingSum', 'Year', 'Month', 'Day', 'WeekDay', 'barcode', 'price',
                    'TotalCount', 'isDiscounted', 'discountAmount', 'totalSum', 'ProductTypeClean', 'name',
                    'NameTypeText', 'Category']
        target_var = 'Hour'
        self.x_train, self.y_train, self.x_test, self.y_test = train[features], train[target_var], test[features], test[
            target_var]

    def encoding(self):
        features_to_encode = ['CustomerId', 'isDiscounted', 'ProductTypeClean', 'name', 'NameTypeText', 'Category']
        l_encoder = LabelEncoder()
        self.inverse_mapping = {}
        for feature in features_to_encode:
            l_encoder.fit(self.cabinet_df[feature])
            self.x_train[feature] = l_encoder.transform(self.x_train[feature])
            self.x_test[feature] = l_encoder.transform(self.x_test[feature])
            self.inverse_mapping[feature] = dict(zip(l_encoder.transform(l_encoder.classes_), l_encoder.classes_))

            # Observe that this fixes pandas produces
            # x_train.loc[:, feature] = l_encoder.transform(x_train.loc[:, feature]) #  test
            # x_test.loc[:, feature] = l_encoder.transform(x_test.loc[:, feature]) #  test

    def train_predict(self):
        self.model.fit(self.x_train, self.y_train)
        self.predictions = self.model.predict(self.x_test)

        # stacked probability table
        self.predictions_proba = self.model.predict_proba(self.x_test)

    def aggregate_probabilities(self):
        self.predictions_proba = pd.DataFrame(self.predictions_proba) if self.model_name == 'MLP' else pd.DataFrame(
            self.predictions_proba,
            columns=self.model.classes_)
        self.x_test['predictions'] = self.predictions
        self.x_test['actuals'] = self.y_test

    def evaluate(self):
        #  Adding scores
        hit_score = eval_funcs.compute_hit_score(self.predictions_proba, self.y_test, top_n=5)
        self.x_test['hit_score'] = hit_score

        score = accuracy_score(self.predictions, self.y_test)
        self.x_test['accuracy_score'] = score

    def build_output(self):
        agg_x_test = self.x_test.reset_index(drop=True).join(self.predictions_proba)
        agg_x_test['name'] = agg_x_test['name'].map(self.inverse_mapping['name'])

        # Additional aggregation of probabilities
        agg_x_test['DeviceName'] = self.cab_name
        agg_x_test['model_name'] = self.model_name

        agg_x_test = agg_x_test.melt(
            id_vars=['DeviceName', 'name', 'barcode', 'model_name', 'accuracy_score', 'hit_score'],
            value_vars=list(self.model.classes_), var_name='Hour',
            value_name='sales_probabilities',
            col_level=None,
            ignore_index=True).sort_values(
            ['DeviceName', 'barcode']).reset_index(drop=True)
        return agg_x_test

    def run_steps(self):
        logging.info('RUNNING PREPROCESSING')
        self.preprocessing()
        logging.info('PERFORMING TRAIN TEST SPLIT')
        self.train_test_split()
        logging.info('PERFORMING ENCODING')
        self.encoding()
        logging.info('PERFORMING TRAIN PREDICT')
        self.train_predict()
        logging.info('AGGREGATING PROBABILITIES')
        self.aggregate_probabilities()
        logging.info('RUNNING EVALUATION')
        self.evaluate()
        logging.info('BUILDING OUTPUT')
        agg_x_test = self.build_output()
        return agg_x_test



