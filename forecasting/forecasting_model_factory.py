from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
from sklearn import ensemble as ensemble
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense, Dropout, Attention, Input, Layer
from tensorflow.keras.models import Model, Sequential
from keras.callbacks import EarlyStopping
import keras.backend as K
from xgboost import XGBRegressor
# from pmdarima.arima import auto_arima

# Local Libraries
import md_helpers as helpers
from forecasting.modelling_pipelines import ForecastingAPipe

pd.options.mode.chained_assignment = None


class attention(Layer):
    def __init__(self, **kwargs):
        super(attention, self).__init__(**kwargs)
        self.b = None
        self.W = None

    def build(self, input_shape):
        self.W = self.add_weight(name='attention_weight', shape=(input_shape[-1], 1),
                                 initializer='random_normal', trainable=True)
        self.b = self.add_weight(name='attention_bias', shape=(input_shape[1], 1),
                                 initializer='zeros', trainable=True)
        super(attention, self).build(input_shape)

    def call(self, x):
        # Alignment scores. Pass them through thanage function
        e = K.tanh(K.dot(x, self.W) + self.b)
        # Remove dimension of size 1
        e = K.squeeze(e, axis=-1)
        # Compute the weights
        alpha = K.softmax(e)
        # Reshape to tensorFlow format
        alpha = K.expand_dims(alpha, axis=-1)
        # Compute the context vector
        context = x * alpha
        context = K.sum(context, axis=1)
        return context

class MultiLSTMAttention(ForecastingAPipe):

    def __init__(self, resample_method, train_size, infer_mode, use_cv_score=False, sliding_window=14,
                 loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'], epochs=10,
                 batch_size=64, n_days_predicted=7, weather_df=None, holidays_df=None):
        super().__init__(resample_method, train_size, weather_df=weather_df, holidays_df=holidays_df)

        self.x_y_full = None
        self.y_scaled = None
        self.x_reshaped = None
        self.splitter = None
        self.test_full = None
        self.train_full = None
        self.x_train_reshaped = None
        self.y_test = None
        self.x_test = None
        self.y_train = None
        self.infer_mode = infer_mode
        self.x_scaler = MinMaxScaler()
        self.y_scaler = MinMaxScaler()
        self.use_cv_score = use_cv_score
        self.sliding_window = sliding_window
        self.loss = loss
        self.optimizer = optimizer
        self.metrics = metrics
        self.epochs = epochs
        self.batch_size = batch_size
        self.n_days_predicted = n_days_predicted

    def setter(self, barcode, cabinet_df):
        self.barcode = barcode
        self.cabinet_df = cabinet_df

    def train_test_split(self):

        self.splitter = helpers.SplitsHandler(self.resample_method)
        if not self.infer_mode:
            self.x_train, self.y_train, self.x_test, self.y_test = self.splitter.train_test_split(
                self.timeseries_dataframe,
                train_size=self.train_size
            )
            self.scale_reshape_operation_train()
        else:
            self.x, self.y = self.splitter.train_test_split(self.timeseries_dataframe,
                                                            train_size=self.train_size,
                                                            split_method='x_y'
                                                            )
            self.scale_reshape_operation_infer()

    @staticmethod
    def generate_full_dataset(x, y):
        full_dataset = x.copy()
        full_dataset['TotalCount'] = y

        return full_dataset

    def scale_reshape_operation_train(self):
        self.train_full = self.generate_full_dataset(self.x_train, self.y_train)
        self.test_full = self.generate_full_dataset(self.x_test, self.y_test)

        # Scales the training and testing sets using the x_scaler and y_scaler.
        x_scaled = self.x_scaler.fit_transform(self.train_full)
        y_scaled = self.y_scaler.fit_transform(self.train_full[['TotalCount']])

        # 2. Reshapes the training set into a 3-dimensional tensor suitable for input into a LSTM model using
        # multi_lstm_splits method from the helpers.SplitsHandler class.
        x_train, self.y_train = self.splitter.multi_lstm_splits(x_scaled,
                                                                y_scaled,
                                                                sliding_window=self.sliding_window)
        # 3. Reshapes the training set into a 3-dimensional tensor suitable for input into a LSTM model using
        # multi_lstm_splits method from the helpers.SplitsHandler class.
        self.x_train_reshaped = np.reshape(x_train,
                                           (x_train.shape[0], x_train.shape[1], x_train.shape[2]))

    def scale_reshape_operation_infer(self):
        self.x_y_full = self.generate_full_dataset(self.x, self.y)

        # Scales the training and testing sets using the x_scaler and y_scaler.
        x_scaled = self.x_scaler.fit_transform(self.x_y_full)
        y_scaled = self.y_scaler.fit_transform(self.x_y_full[['TotalCount']])

        # 3. Reshapes the training set into a 3-dimensional tensor suitable for input into a LSTM model using
        # multi_lstm_splits method from the helpers.SplitsHandler class.

        x_scaled, self.y_scaled = self.splitter.multi_lstm_splits(x_scaled,
                                                                  y_scaled,
                                                                  sliding_window=self.sliding_window)
        # 4. Reshapes the training set into a 3-dimensional tensor suitable for input into a LSTM model using
        # multi_lstm_splits method from the helpers.SplitsHandler class.
        self.x_reshaped = np.reshape(x_scaled,
                                     (x_scaled.shape[0], x_scaled.shape[1], x_scaled.shape[2]))

    def build_lstm(self):
        # Define n_features based on the shape of self.test_full
        n_features = self.test_full.shape[1] if not self.infer_mode else self.x_y_full.shape[1]

        # Define the inputs for the model
        inputs = Input(shape=(self.sliding_window, n_features))

        # Define the LSTM layer with return sequences set to True
        lstm_1 = LSTM(32, return_sequences=True)(inputs)
        dropout_1 = Dropout(0.2)(lstm_1)

        # Define the Attention layer
        attention_inputs = [lstm_1, lstm_1, lstm_1]
        attention = Attention()(attention_inputs)

        # Define the remaining LSTM layers
        lstm_2 = LSTM(32, return_sequences=True)(attention)
        dropout_2 = Dropout(0.2)(lstm_2)

        lstm_3 = LSTM(32)(dropout_2)
        dropout_3 = Dropout(0.2)(lstm_3)

        # Define the output layer
        outputs = Dense(1)(dropout_3)

        # Define the model
        model = Model(inputs, outputs)

        # Compile the model
        model.compile(loss=self.loss, optimizer=self.optimizer, metrics=self.metrics)

        return model

    def build_lstm_custom_attention_layer(self):
        # Define the input shape based on the shape of self.test_full or self.x_y_full
        n_features = self.test_full.shape[1] if not self.infer_mode else self.x_y_full.shape[1]
        input_shape = (self.sliding_window, n_features)

        # Define the inputs for the model
        inputs = Input(shape=input_shape)

        # Define the LSTM layers with return sequences set to True
        lstm_1 = LSTM(32, return_sequences=True)(inputs)

        # Define the Attention layer
        attention_layer = attention()(lstm_1)

        # Define the output layer
        outputs = Dense(1)(attention_layer)

        # Define the model
        model = Model(inputs, outputs)

        # Compile the model
        model.compile(loss=self.loss, optimizer=self.optimizer, metrics=self.metrics)

        return model

    def train_predict(self):
        # 1. Build the model
        model = self.build_lstm()

        # 2. Fit on self.x_train_reshaped, self.y_train
        early_stop = EarlyStopping(monitor=self.loss, mode='min', patience=5)

        model.fit(self.x_train_reshaped, self.y_train, epochs=self.epochs, batch_size=self.batch_size,
                  callbacks=[early_stop], verbose=0)

        # 3. Selects the last self.sliding_window days from self.train_full to use as the initial data for prediction.
        last_n_days = self.train_full.iloc[-self.sliding_window:]

        # 4. Concatenates the last self.sliding_window days with self.test_full to form a full test set.
        full_test = pd.concat([last_n_days, self.test_full], axis=0)

        # 5. Transforms full_test using self.x_scaler. Forms LSTM-friendly input data for the prediction using the
        # multi_lstm_splits method with the full_test data and the self.y_test data, using a sliding window of
        # self.sliding_window and setting the return_only_test flag to True.
        full_test = self.x_scaler.transform(full_test)

        custom_x_test = self.splitter.multi_lstm_splits(full_test,
                                                        self.y_test,
                                                        sliding_window=self.sliding_window,
                                                        return_only_test=True)

        # 6. Uses the fitted model to make predictions on the LSTM-friendly input data.
        predictions = model.predict(custom_x_test)
        # 7. Transforms the predictions back to their original scale using self.y_scaler.inverse_transform.
        predictions = self.y_scaler.inverse_transform(predictions)
        # 8. Converts the predictions to a pandas Series and stores the result in self.predictions.
        predictions = pd.Series(predictions.flatten())
        predictions = [max(0, y) for y in predictions]
        predictions = pd.Series(predictions)

        return predictions

    def infer_predict(self):
        # 1. Build the model
        model = self.build_lstm()

        # 2. Fit on self.x_train_reshaped, self.y_train
        early_stop = EarlyStopping(monitor=self.loss, mode='min', patience=5)

        infer_dataset = helpers.generate_future_dataset(self.x,
                                                        n_days_predicted=self.n_days_predicted,
                                                        resample_method=self.resample_method,
                                                        cabinet_df=self.cabinet_df,
                                                        weather_df=self.weather_df,
                                                        holidays_df=self.holidays_df
                                                        )
        infer_dataset['TotalCount'] = 0

        model.fit(self.x_reshaped, self.y_scaled, epochs=self.epochs, batch_size=self.batch_size,
                  callbacks=[early_stop], verbose=0)

        # 3. Selects the last self.sliding_window days from self.train_full to use as the initial data for prediction.
        last_n_days = self.x_y_full.iloc[-self.sliding_window:]

        # 4. Concatenates the last self.sliding_window days with self.test_full to form a full test set.
        full_test = pd.concat([last_n_days, infer_dataset], axis=0)

        # 5. Transforms full_test using self.x_scaler. Forms LSTM-friendly input data for the prediction using the
        # multi_lstm_splits method with the full_test data and the self.y_test data, using a sliding window of
        # self.sliding_window and setting the return_only_test flag to True.
        full_test = self.x_scaler.transform(full_test)

        custom_x_test = self.splitter.multi_lstm_splits(full_test,
                                                        self.y_test,
                                                        sliding_window=self.sliding_window,
                                                        return_only_test=True)

        # 6. Uses the fitted model to make predictions on the LSTM-friendly input data.
        predictions = model.predict(custom_x_test)
        # 7. Transforms the predictions back to their original scale using self.y_scaler.inverse_transform.
        predictions = self.y_scaler.inverse_transform(predictions)
        # 8. Converts the predictions to a pandas Series and stores the result in self.predictions.
        predictions = pd.Series(predictions.flatten())
        predictions = [max(0, y) for y in predictions]

        infer_dataset['actuals'] = predictions
        return infer_dataset


class Ensembles(ForecastingAPipe):

    def __init__(self, model_name, resample_method, train_size, infer_mode, use_oversampling=False,
                 use_cv_score=False, n_days_predicted=7, weather_df=None, holidays_df=None):
        super().__init__(resample_method, train_size, infer_mode,  weather_df=weather_df,
                         holidays_df=holidays_df)
        self.y = None
        self.x = None
        self.model_name = model_name
        self.use_cv_score = use_cv_score
        self.model = None
        self.use_oversampling = use_oversampling
        self.n_days_predicted = n_days_predicted

    def setter(self, barcode, cabinet_df):
        self.barcode = barcode
        self.cabinet_df = cabinet_df

    def train_test_split(self):
        splitter = helpers.SplitsHandler(self.resample_method)

        if not self.infer_mode:
            self.x_train, self.y_train, self.x_test, self.y_test = splitter.train_test_split(self.timeseries_dataframe,
                                                                                             train_size=self.train_size)
        else:
            self.x, self.y = splitter.train_test_split(self.timeseries_dataframe,
                                                       train_size=self.train_size,
                                                       split_method='x_y'
                                                       )

    def build_ensemble(self):
        if self.model_name == 'RF':
            self.model = ensemble.RandomForestRegressor(random_state=self.random_state)
        elif self.model_name == 'XGB':
            self.model = XGBRegressor(random_state=self.random_state)
        else:
            raise ValueError('Invalid model')

        return self.model

    def train_predict(self):

        # Train and return predictions
        ensemble_model = self.build_ensemble()

        ensemble_model.fit(self.x_train, self.y_train)
        self.predictions = ensemble_model.predict(self.x_test)

        # In the case of xgboost it uses gradient boosting and this can produce negative outputs
        self.predictions = [max(0, y) for y in self.predictions]

        return self.predictions

    def infer_predict(self):
        # 1. Define splits
        infer_dataset = helpers.generate_future_dataset(self.x,
                                                        n_days_predicted=self.n_days_predicted,
                                                        resample_method=self.resample_method,
                                                        cabinet_df=self.cabinet_df,
                                                        weather_df=self.weather_df,
                                                        holidays_df=self.holidays_df
                                                        )

        # 2. Train and return predictions
        ensemble_model = self.build_ensemble()
        # ------------------------
        # self.x should be normalized
        # ------------------------
        ensemble_model.fit(self.x, self.y)
        self.predictions = ensemble_model.predict(infer_dataset)

        # In the case of xgboost it uses gradient boosting and this can produce negative outputs
        self.predictions = [max(0, y) for y in self.predictions]

        infer_dataset['actuals'] = self.predictions
        return infer_dataset


class MLPClassifier:
    def __init__(self, epochs, verbose, batch_size):  # early_stop
        self.mlp_model = None
        self.epochs = epochs
        self.verbose = verbose
        # self.early_stop = early_stop
        self.batch_size = batch_size
        self.classes_ = None

    @staticmethod
    def build_mlp_classifier(y_train_n):
        n_classes = y_train_n
        model = Sequential()
        model.add(Dense(32, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(32, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(n_classes, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam')
        return model

    def fit(self, x_train, y_train):
        self.classes_ = y_train.unique()
        y_train = tf.keras.utils.to_categorical(y_train)
        self.mlp_model = self.build_mlp_classifier(y_train.shape[1])

        # Perform scaling to x_train
        mm_scaler = MinMaxScaler()
        x_train_scaled = mm_scaler.fit_transform(x_train)
        self.mlp_model.fit(x_train_scaled, y_train, epochs=self.epochs, verbose=self.verbose,
                           batch_size=self.batch_size)  # callbacks=[self.early_stop]

    def predict(self, x_test):
        if self.mlp_model:
            # Perform scaling to x_test
            mm_scaler = MinMaxScaler()
            x_test_scaled = mm_scaler.fit_transform(x_test)
            prediction = self.mlp_model.predict(x_test_scaled, batch_size=self.batch_size, verbose=0)
            prediction = np.argmax(prediction, axis=1)
            return prediction
        else:
            raise ValueError('Fit the model first')

    def predict_proba(self, x_test):
        if self.mlp_model:
            # Perform scaling to x_tes
            mm_scaler = MinMaxScaler()
            x_test_scaled = mm_scaler.fit_transform(x_test)
            prediction = self.mlp_model.predict(x_test_scaled, batch_size=self.batch_size, verbose=0)
            return prediction
        else:
            raise ValueError('Fit the model first')

# --------------
#  APPENDIX CODE
# --------------


# # OBSERVE THAT IF WE PLAN TO USE THIS MODEL WEATHER PARAMETER SHOULD BE ADDED TO THE OBJECT
# class PMDARIMAlgo(ForecastingAPipe):
#     def __init__(self, resample_method, train_size,
#                  barcode=None, cabinet_df=None, is_classifier=False):
#         super().__init__(resample_method, train_size, barcode, cabinet_df, is_classifier)
#         self.model = None
#         self.use_cv_score = None
#         self.splitter = None
#
#     def setter(self, barcode, cabinet_df):
#         self.barcode = barcode
#         self.cabinet_df = cabinet_df
#
# def train_test_split(self): self.splitter = helpers.SplitsHandler(self.resample_method) self.x_train, self.y_train,
# self.x_test, self.y_test = self.splitter.train_test_split(self.timeseries_dataframe, train_size=self.train_size )
#
#     # TODO: adapt m parameters dynamically
#     # def arima_m_setter(resample_method):
#     #     m = None
#     #     if resample_method == 'H':
#     #         m = 24
#     #     elif resample_method in ['d', 'D']:
#     #         m = 12
#     #
#     #     return m
#
#     #     def build_arima(self):
#     #
#     # m = arima_m_setter(self.resample_method) stepwise_fit = auto_arima(self.splits_container['train_data'],
#     # start_p=1, start_q=1, max_p=3, max_q=3, m=m, seasonal=True, stepwise=True, suppress_warnings=True) model =
#     # ARIMA(self.splits_container['train_data'], order=stepwise_fit.order) return model
#     #
#
#     def build_pmd_arima(self):
#         model = auto_arima(self.y_train, start_p=0, start_q=0,
#                            max_p=7, max_q=7, m=7,
#                            start_P=0, seasonal=True,
#                            d=1, D=1, trace=True,
#                            error_action='ignore',
#                            suppress_warnings=True,
#                            stepwise=True)
#         return model
#
#     def train_predict(self):
#         self.model = self.build_pmd_arima()
#         self.model.fit(self.y_train)
#         self.predictions = self.model.predict(len(self.y_test))
#         self.predictions[self.predictions < 0] = 0
#         # Observe that the predictions should be converted to pd.Series and currently we have an error
#         self.predictions = pd.Series(self.predictions)
#         return self.predictions
#
#     # def evaluate(self):
#     #     errors_hash = eval_funcs.aggregate_mse_mae(self.model,
#     #                                                self.predictions,
#     #                                                self.y_test,
#     #                                                self.timeseries_dataframe,
#     #                                                self.resample_method,
#     #                                                use_cv_score=self.use_cv_score,
#     #
#     #                                                )
#     #     return errors_hash
#


# # --------------------------------------------------------------------------------------------
# # TODO: THIS BLOCK SHOULD BE MOVED TO A SEPARATE PLACE
# class CustomEarlyStopping(tf.keras.callbacks.Callback):
#     def on_epoch_end(self, epoch, logs={}):
#         if logs.get('mae') < 0.03:
#             print("\nMAEthreshold reached. Training stopped.")
#             self.model.stop_training = True
#
#
# # This is another way of implementing early stopping
# # early_stop = EarlyStopping(monitor=metric_to_monitor, mode=mode, verbose=verbose, patience=patience)
# # --------------------------------------------------------------------------------------------
#

# # OBSERVE THAT IF WE PLAN TO USE THIS MODEL WEATHER PARAMETER SHOULD BE ADDED TO THE OBJECT
# # OBSERVE THAT THIS MODEL IS NOT IN USE AND THEREFORE WE ARE NOT MAINTAINING IT
# class UniLSTM(ForecastingAPipe):
#
#     def __init__(self, resample_method, train_size, barcode=None, cabinet_df=None,
#                  is_classifier=False, sliding_window=14, loss=tf.keras.losses.Huber(),
#                  optimizer=tf.keras.optimizers.Nadam(learning_rate=0.001), metrics=["mae"], epochs=10, batch_size=64):
#         super().__init__(resample_method, train_size, barcode, cabinet_df, is_classifier)
#         self.model = None
#         self.use_cv_score = None
#         self.splitter = None
#         self.scaler = None
#         self.sliding_window = sliding_window
#         self.loss = loss
#         self.optimizer = optimizer
#         self.metrics = metrics
#         self.epochs = epochs
#         self.batch_size = batch_size
#
#     def setter(self, barcode, cabinet_df):
#         self.barcode = barcode
#         self.cabinet_df = cabinet_df
#
#     # TODO: The scaler should be removed from the output
#     def train_test_split(self):
#         self.splitter = helpers.SplitsHandler(self.resample_method)
#         self.x_train, self.y_train, self.x_test, self.y_test, self.scaler = self.splitter.uni_lstm_splits(
#             self.timeseries_dataframe,
#             train_size=self.train_size,
#             sliding_window=self.sliding_window)
#
#     def build_lstm(self):
#         model = tf.keras.models.Sequential([
#             tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1),
#                                    input_shape=[None]),
#             tf.keras.layers.LSTM(32, return_sequences=True),
#             tf.keras.layers.Dropout(0.1),
#             tf.keras.layers.LSTM(32, return_sequences=True),
#             tf.keras.layers.Dropout(0.1),
#             tf.keras.layers.LSTM(32),
#             tf.keras.layers.Dropout(0.1),
#             tf.keras.layers.Dense(1),
#         ])
#         # Compile the model
#         model.compile(loss=self.loss,
#                       optimizer=self.optimizer,
#                       metrics=self.metrics)
#
#         return model
#
#     def train_predict(self):
#         self.model = self.build_lstm()
#         early_stopping = CustomEarlyStopping()
#         self.model.fit(self.x_train, self.y_train, epochs=self.epochs, batch_size=self.batch_size,
#                        callbacks=[early_stopping])
#         self.predictions = self.model.predict(self.x_test)
#
#         self.predictions = pd.Series(self.predictions.flatten())
#         # Observe that the predictions should be converted to pd.Series and currently we have an error
#         self.predictions[self.predictions < 0] = 0
#         return self.predictions
#
#     # def evaluate(self): errors_hash = eval_funcs.aggregate_mse_mae(self.model, self.predictions,
#     # self.scaler.inverse_transform(self.y_test.reshape(-1, 1)).flatten(), self.timeseries_dataframe,
#     # self.resample_method, use_cv_score=self.use_cv_score,
#     #
#     #                                                )
#     #     return errors_hash
#
#     def build_output(self):
#         dataset_builder: DatasetFactory = DatasetFactory()
#
#         train_output_dataset = dataset_builder.build_lstm_test_dataset(self.predictions,
#                                                                        self.x_test,
#                                                                        self.scaler.inverse_transform(
#                                                                            self.y_test.reshape(-1, 1)).flatten(),
#                                                                        self.timeseries_dataframe,
#                                                                        self.resample_method)
#
#         return train_output_dataset
#
#
# # OBSERVE THAT IF WE PLAN TO USE THIS MODEL WEATHER PARAMETER SHOULD BE ADDED TO THE OBJECT
# class BidirectionalAlgo(ForecastingAPipe):
#
#     def __init__(self, resample_method, train_size, barcode=None, cabinet_df=None, is_classifier=False,
#                  sliding_window=14,
#                  loss=tf.keras.losses.Huber(), optimizer=tf.keras.optimizers.Nadam(learning_rate=0.001),
#                  metrics=["mae"],
#                  epochs=10, batch_size=64
#                  ):
#         super().__init__(resample_method, train_size, barcode, cabinet_df, is_classifier)
#         self.model = None
#         self.use_cv_score = None
#         self.splitter = None
#         self.sliding_window = sliding_window
#         self.loss = loss
#         self.optimizer = optimizer
#         self.metrics = metrics
#         self.epochs = epochs
#         self.batch_size = batch_size
#
#     def setter(self, barcode, cabinet_df):
#         self.barcode = barcode
#         self.cabinet_df = cabinet_df
#
#     def train_test_split(self):
#         self.splitter = helpers.SplitsHandler(self.resample_method)
#         self.x_train, self.y_train, self.x_test, self.y_test, self.scaler = self.splitter.uni_lstm_splits(
#             self.timeseries_dataframe,
#             train_size=self.train_size,
#             sliding_window=self.sliding_window)
#
#     def build_bid_lstm(self):
#         model = tf.keras.models.Sequential([
#             tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1),
#                                    input_shape=[None]),
#             tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(1024, return_sequences=True)),
#             tf.keras.layers.Dropout(0.1),
#             tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(512, return_sequences=True)),
#             tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256, return_sequences=True)),
#             tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True)),
#             tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
#             tf.keras.layers.Dense(1), ])
#
#         tf.random.set_seed(51)
#         model.compile(loss=self.loss,
#                       optimizer=self.optimizer,
#                       metrics=self.metrics)
#         return model
#
#     def train_predict(self):
#         self.model = self.build_bid_lstm()
#         early_stopping = CustomEarlyStopping()
#         self.model.fit(self.x_train, self.y_train, epochs=self.epochs, batch_size=self.batch_size,
#                        callbacks=[early_stopping])
#         self.predictions = self.model.predict(self.x_test)
#         # self.predictions = self.scaler.inverse_transform(self.predictions)
#         # Observe that the predictions should be converted to pd.Series and currently we have an error
#         self.predictions = pd.Series(self.predictions.flatten())
#         # self.predictions[self.predictions<0] = 0
#         return self.predictions
#
#     # def evaluate(self): errors_hash = eval_funcs.aggregate_mse_mae(self.model, self.predictions,
#     # self.scaler.inverse_transform(self.y_test.reshape(-1, 1)).flatten(), self.timeseries_dataframe,
#     # self.resample_method, use_cv_score=self.use_cv_score,
#     #
#     #                                                )
#     #     return errors_hash
#
#     def build_output(self):
#         dataset_builder: DatasetFactory = DatasetFactory()
#         train_output_dataset = dataset_builder.build_lstm_test_dataset(self.predictions,
#                                                                        self.x_test,
#                                                                        self.scaler.inverse_transform(
#                                                                            self.y_test.reshape(-1, 1)).flatten(),
#                                                                        self.timeseries_dataframe,
#                                                                        self.resample_method)
#
#         return train_output_dataset


# class BidirectionalAlgo(AIAlgorithm):
#
#     def __init__(self, resample_method, train_size, use_oversampling, random_state=0,
#                  barcode=None, cabinet=None, classifier=False):
#         super().__init__(resample_method, train_size, use_oversampling, random_state, barcode, cabinet, classifier)
#         self.model = None
#         self.use_cv_score = None
#         self.splitter = None
#
#     def setter(self, barcode, cabinet):
#         self.barcode = barcode
#         self.cabinet = cabinet
#
# def train_test_split(self): self.splitter = helpers.SplitsHandler(self.resample_method) self.x_train, self.y_train,
# self.x_test, self.y_test = self.splitter.train_test_split(self.timeseries_dataframe, train_size=self.train_size,
# use_oversampling=self.use_oversampling)
#
#     @staticmethod
#     def build_bid_lstm():
#         model = tf.keras.models.Sequential([
#             tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1),
#                                    input_shape=[None]),
#             tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(1024, return_sequences=True)),
#             tf.keras.layers.Dropout(0.2),
#             tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(512, return_sequences=True)),
#             tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256, return_sequences=True)),
#             tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True)),
#             tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
#             tf.keras.layers.Dense(1), ])
#
#         tf.random.set_seed(51)
#         model.compile(loss=tf.keras.losses.Huber(),
#                       optimizer=tf.keras.optimizers.Nadam(learning_rate=0.001),
#                       metrics=["mae"])
#         return model
#
#     def train_predict(self):
#         self.model = self.build_bid_lstm()
#         early_stopping = EarlyStopping()
#         self.model.fit(self.x_train, self.y_train, epochs=2, batch_size=64, callbacks=[early_stopping])
#         self.predictions = self.model.predict(self.x_test)
#         # Observe that the predictions should be converted to pd.Series and currently we have an error
#         self.predictions = pd.Series(self.predictions.flatten())  # TESTING
#         return self.predictions
#
#     def evaluate(self):
#         errors_hash = aggregate_mse_mae(self.model,
#                                                self.predictions,
#                                                self.y_test,
#                                                self.timeseries_dataframe,
#                                                self.resample_method,
#                                                use_cv_score=self.use_cv_score
#                                                )
#         return errors_hash
