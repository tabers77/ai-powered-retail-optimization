import md_helpers
import md_utils as ut
import md_evaluators as eval_funcs
from forecasting.modelling_pipelines import ForecastingBPipe
import pandas as pd
from tqdm import tqdm
from time import time
from global_preprocessor import GlobalPreprocessor

def output_df_post_processing(predicted_df, df, infer_mode=False):
    """
    Post-processes the predicted DataFrame and returns the desired output DataFrame.

    Args:
        predicted_df (pd.DataFrame): Predicted DataFrame.
        df (pd.DataFrame): Original DataFrame.
        infer_mode (bool): Flag indicating if the function is called in inference mode.

    Returns:
        pd.DataFrame: Output DataFrame.

    """
    if infer_mode:
        predicted_df['is_predicted'] = predicted_df['is_predicted'].apply(lambda x: 'FALSE' if x != 'TRUE' else x)

    if 'TotalCount' in predicted_df.columns:
        predicted_df.drop('TotalCount', axis=1, inplace=True)
    predicted_df.reset_index(inplace=True)

    output_df = pd.merge(predicted_df,
                         df[['DeviceName', 'barcode', 'name']].drop_duplicates(subset=['DeviceName', 'barcode']),
                         how='inner', on=['DeviceName', 'barcode'])
    if infer_mode:
        output_df = output_df[['datetime', 'Year', 'week', 'DeviceName','Organization_Id', 'barcode', 'name', 'actuals', 'is_predicted',
                               'model_name']]

    else:
        output_df = output_df[['datetime', 'Year', 'week', 'DeviceName','Organization_Id', 'barcode', 'name', 'actuals',
                               'predictions','model_name', 'mean_squared_error', 'root_mean_squared_error', 'mape',
                               'confidence_label']]
    return output_df


# TODO: IMPLEMENT CHANGES ADDED TO INIT
class Compilers:
    """
    Class for performing compilation operations.

    Args:
        df (pd.DataFrame): Input DataFrame.
        n_days_limit (int): Number of days limit.
        n_pct_cabinets (float): Percentage of cabinets.
        train_size (float): Training size.
        sparsity_level (float): Sparsity level.
        infer_mode (bool): Flag indicating if the class is in inference mode.
        start_date (str): Start date.
        orgs_ids_list (list): List of organization IDs.

    """

    def __init__(self, df, n_days_limit, n_pct_cabinets, train_size=0.85, sparsity_level=0.75, infer_mode=False, start_date='2021-06-01',
                 orgs_ids_list=None):
        self.df = df
        self.n_days_limit = n_days_limit
        self.n_pct_cabinets = n_pct_cabinets
        self.train_size = train_size
        self.sparsity_level = sparsity_level
        self.infer_mode = infer_mode
        self.start_date = start_date
        self.orgs_ids_list = orgs_ids_list

        # ------------------------
        # Initiate initial filters
        # ------------------------
        print ('Running initial preprocessors...')

        # FILTER 1: Obtain TOP N cabinets
        # self.copy_df = md_helpers.preprocessor1_df_level(self.df, start_date=self.start_date)
        # TEST

        glob_preprocessor = GlobalPreprocessor(df= self.df, start_date =self.start_date, orgs_ids_list=self.orgs_ids_list)
        self.copy_df = glob_preprocessor.run_preprocessor()

        self.org_id_hash_map = dict(zip(self.copy_df.DeviceName, self.copy_df.Organization_Id))
        # FILTER 2: Generate top cabs
        self.top_cabs = md_helpers.get_top_cabs_based_on_barcodes(self.copy_df,
                                                                  n_pct_cabinets=self.n_pct_cabinets,
                                                                  n_days_limit=self.n_days_limit
                                                                  )

    def add_extra_features_to_x_predicted(self,cab_name,barcode,model_name, x_predicted,errors_hash=None,
                                          train_outputs=True):

        x_predicted['DeviceName'] = cab_name
        x_predicted['Organization_Id'] = x_predicted['DeviceName'].map(self.org_id_hash_map)
        x_predicted['barcode'] = barcode
        x_predicted['model_name'] = model_name
        if train_outputs:
            x_predicted['mean_squared_error'] = errors_hash['mse']
            x_predicted['root_mean_squared_error'] = errors_hash['rmse']
            x_predicted['mape'] = errors_hash['mape']
            x_predicted['confidence_label'] = x_predicted['mape'].apply(
                lambda x: ut.get_mapes_confidence_label(x))

        return x_predicted

    def train_compiler(self, models):

        # Initialize the containers
        comparison = dict()
        dataframes = []

        for model_name, model in models:
            for cab_name in tqdm(self.top_cabs, desc=f'{model_name}'):
                # ----------------------------------
                # FILTER 3: Define cabinet df
                # ----------------------------------
                cabinet_df = md_helpers.preprocessor2_cabinet_level(self.copy_df, cab_name)
                # -----------------------
                # 2. FILTER 4: Obtain TOP barcodes
                # -----------------------

                top_barcodes = md_helpers.get_top_barcodes_wrapper(cabinet_df= cabinet_df,
                                                                   n_days_limit= self.n_days_limit,
                                                                   train_size=self.train_size,
                                                                   sparsity_level=self.sparsity_level)

                print(f'Number of top barcodes for cabinet: {cab_name}: {len(top_barcodes)}')
                # -----------------------------
                # 3. Loop through model options
                # -----------------------------
                error_container = dict()
                for barcode in top_barcodes:
                    model.setter(barcode, cabinet_df)
                    # Error aggregation 1: Error has returns: errors_hash['mse'] = mse
                    t1 = time()
                    x_predicted, errors_hash = model.run()
                    t2 = time()
                    time_total = t2 - t1
                    errors_hash['time_total'] = time_total

                    # Error aggregation 2: returns {'mse':[1,2,3,4], 'mae':[1,2,3,4] } if we have 4 barcodes --> 1,2,3,4
                    eval_funcs.s1_collect_errors_from_error_hash(error_container, errors_hash)
                    x_predicted = self.add_extra_features_to_x_predicted(cab_name,barcode,model_name,x_predicted,
                                                                         errors_hash)

                    dataframes.append(x_predicted)

                # Error aggregation 3:
                comparison, mean_agg, std_agg = eval_funcs.s2_aggregate_individual_errors(model_name, cab_name,
                                                                                          comparison,
                                                                                          error_container)

        # Error aggregation 4: Get an aggregation error for all barcodes
        summary_results_table = eval_funcs.s3_aggregate_all_level_values(comparison)
        # MLFLOW AGGREGATION: self.n_pct_cabinets, self.n_days_limit, self.windows_size, self.resample_method,
        #                                   model, model_name, cv_errors, mse_errors

        output_df = output_df_post_processing(predicted_df=pd.concat(dataframes), df=self.copy_df, infer_mode=self.infer_mode)
        return output_df, summary_results_table

    def infer_compiler(self, models:dict, dataset_train:pd.DataFrame):
        best_model_hashmap = md_helpers.get_best_model(dataset_train)
        # Initialize the containers
        dataframes = []
        for cab_name in tqdm(self.top_cabs):
            # ----------------------------------
            # FILTER 3: Define cabinet df
            # ----------------------------------
            cabinet_df = md_helpers.preprocessor2_cabinet_level(self.copy_df, cab_name)
            # -----------------------
            # 2. FILTER 4: Obtain TOP barcodes
            # -----------------------

            top_barcodes = md_helpers.get_top_barcodes_wrapper(cabinet_df=cabinet_df,
                                                               n_days_limit=self.n_days_limit,
                                                               train_size=self.train_size,
                                                               sparsity_level=self.sparsity_level)

            print(f'Number of top barcodes for cabinet: {cab_name}: {len(top_barcodes)}')
            # -----------------------------
            # 3. Loop through model options
            # -----------------------------

            for barcode in top_barcodes:
                # GET MODEL
                best_model = best_model_hashmap[cab_name][barcode]
                model = models[best_model]
                model.setter(barcode, cabinet_df)

                x_predicted = model.run()

                x_predicted = self.add_extra_features_to_x_predicted(cab_name, barcode, best_model, x_predicted,
                                                                     train_outputs=False)

                dataframes.append(x_predicted)


        output_df = output_df_post_processing(predicted_df=pd.concat(dataframes), df=self.copy_df, infer_mode=self.infer_mode)
        return output_df

    def compiler_forecasting2(self, models, train_size=0.80):

        # Initialize the containers
        dataframes = []
        # ----------------------
        # 1. Obtain TOP N cabinets
        # ----------------------
        # copy_df = prep.preprocessor1_df_level(df)
        # top_cabs = prep.get_top_cabs_based_on_barcodes(copy_df,
        #                                   n_pct_cabinets=self.n_pct_cabinets,
        #                                   n_days_limit=self.n_days_limit)
        for model_name, model in models:
            for cab_name in tqdm(self.top_cabs, desc=f'{model_name}'):
                cabinet_df = md_helpers.preprocessor2_cabinet_level(self.copy_df, cab_name)
                # -----------------------
                # 2. Obtain TOP barcodes
                # -----------------------
                top_barcodes = md_helpers.get_top_barcodes_wrapper(cabinet_df=cabinet_df,
                                                                   n_days_limit=self.n_days_limit,
                                                                   train_size=self.train_size,
                                                                   sparsity_level=self.sparsity_level)

                # -----------------------------
                # 3. Loop through model options
                # -----------------------------

                for barcode in top_barcodes:

                    fb_pipe = ForecastingBPipe(cabinet_df, barcode, train_size, model, model_name, cab_name)
                    x_predicted = fb_pipe.run_steps()

                    dataframes.append(x_predicted)

        output_df = pd.concat(dataframes)
        output_df.fillna(0, inplace=True)

        return output_df

    def run_operation(self, models, dataset_train=None):
        if self.infer_mode:
            return self.infer_compiler(models, dataset_train)
        else:
            return self.train_compiler(models )

#TODO: INTEGRATE THIS FUNCTION TO THE ORIGINAL FUNCTION
# def train_compiler_mlflow_test(self, df, models, train_size=0.80):
#     with mlflow.start_run():
#         comparison = dict()
#         # ----------------------
#         # 1. Obtain TOP N cabinets
#         # ----------------------
#         top_cabs = utils.get_top_cabs_based_on_barcodes(df, n_pct_cabinets=self.n_pct_cabinets, n_days_limit=self.n_days_limit)
#         dataframes = []
#         for cab_name in top_cabs[:1]:  # TEMP LIMIT
#             cabinet_df = prep.preprocessor2_cabinet_level(df, cab_name)
#             # -----------------------
#             # 2. Obtain TOP barcodes
#             # -----------------------
#             top_barcodes = utils.get_top_barcodes(cabinet_df, n_days_limit=self.n_days_limit)
#             print(f'Number of top barcodes for cabinet {cab_name}: {len(top_barcodes)}')
#             # -----------------------------
#             # 3. Loop through model options
#             # -----------------------------
#             for model_name, model in models:
#                 mse_errors = []
#                 cv_errors = []
#                 for barcode in tqdm(top_barcodes, desc=f'{cab_name} {model_name}'):
#                     predictor = TrainersLogic(model_name, self.resample_method)
#                     x_predicted = predictor.main_train_predict_logic(barcode,
#                                                                      mse_errors,
#                                                                      cv_errors,
#                                                                      cabinet_df,
#                                                                      model_name,
#                                                                      train_size=train_size,
#                                                                      resample_method=self.resample_method
#                                                                      )
#
#                     x_predicted['Cabinets'] = cab_name
#                     x_predicted['barcode'] = barcode
#                     x_predicted['model_name'] = model_name
#                     dataframes.append(x_predicted)
#
#                     #  aggregate_mse_mae(cv_errors, model, mse_errors, predictions, timeseries_dataframe,
#                     #  y_test) Get an aggregation error for all barcodes
#                     comparison.setdefault((cab_name, model_name), []).append(
#                         (np.mean(mse_errors), np.mean(cv_errors)))
#
#                 print(f'Error aggregation: {utils.s3_aggregate_all_level_values(comparison)}')
#
#                 # Log the parameters, metrics, and artifacts of the run
#                 mlflow_logger(self.n_pct_cabinets, self.n_days_limit, self.windows_size, self.resample_method,
#                               model, model_name, cv_errors, mse_errors)
#
#                 return comparison, pd.concat(dataframes)
