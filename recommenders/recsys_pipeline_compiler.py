import numpy as np
import pandas as pd
# from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import MinMaxScaler  # ,StandardScaler, QuantileTransformer,
from sklearn.preprocessing import LabelEncoder
import recommenders.recsys_model_factory as models
import recommenders.recsys_helpers as h
from collections import deque
from tqdm import tqdm
import md_helpers
from global_preprocessor import GlobalPreprocessor

# TODO: REPLACE PRINTS WITH LOGS
class Preprocessing:
    def __init__(self, df, n_barcodes_pct=0.30, start_date='2021-06-01', orgs_ids_list=None):
        self.df = df
        self.n_barcodes_pct = n_barcodes_pct
        self.start_date = start_date
        self.orgs_ids_list = orgs_ids_list

    def preprocessing(self):
        # if self.orgs_ids_list is not None:
        #     self.df = self.df[self.df.Organization_Id.isin(self.orgs_ids_list)]
        #
        # copy_df = self.df[self.df.Date >= self.start_date]

        glob_preprocessor = GlobalPreprocessor(df=self.df, start_date=self.start_date, orgs_ids_list=self.orgs_ids_list)
        copy_df = glob_preprocessor.run_preprocessor()

        users_df_base = copy_df[(copy_df.CustomerId.notna()) & (copy_df.CustomerId != 'non_customer_id')]
        users_df_base = users_df_base[(users_df_base.Category != 'uncategorized') & (users_df_base.Status == 'DONE')]

        # REMOVE SOME CABINETS
        users_df_base = users_df_base[users_df_base.cabinet_ranking != 1000000.0]
        return users_df_base

    def pre_filtering(self):
        pre_filtered_df1 = self.preprocessing()
        dataframes = list()

        for cabinet_id in pre_filtered_df1.DeviceName.unique():
            dataframe = pd.DataFrame()

            n_unique_barcodes = pre_filtered_df1[pre_filtered_df1.DeviceName == cabinet_id][
                'barcode'].nunique()

            top_barcodes_pct = int(self.n_barcodes_pct * n_unique_barcodes)

            top_barcodes = pre_filtered_df1[pre_filtered_df1.DeviceName == cabinet_id].groupby('barcode')[
                               'totalSumEuro'].sum().sort_values(ascending=False)[:top_barcodes_pct].keys()

            dataframe['barcode'] = top_barcodes
            dataframe['DeviceName'] = cabinet_id
            dataframes.append(dataframe)

        dataframes = pd.concat(dataframes)
        pre_filtered_df2 = pd.merge(dataframes, pre_filtered_df1, how='left', on=['DeviceName', 'barcode'])
        return pre_filtered_df1, pre_filtered_df2

    def run_steps(self):
        pre_filtered_df1, pre_filtered_df2 = self.pre_filtering()
        return pre_filtered_df1, pre_filtered_df2


class Clustering:
    def __init__(self, pre_filtered_df2):
        self.encoded_df = None
        self.agg_table_scaled = None
        self.agg_table = None
        self.pre_filtered_df2 = pre_filtered_df2

    def generate_aggregated_table(self):
        pre_filtered_df_copy = self.pre_filtered_df2.copy()
        agg_table = pre_filtered_df_copy.groupby(
            ['DeviceName', 'NameClean', 'Category', 'NewLocation', 'locationName', 'Segment_1']).agg(
            {'priceEuro': 'mean', 'TotalCount': 'sum', 'totalSumEuro': 'sum'}).reset_index()

        # --------------
        # Add cab age
        # -------------
        pre_filtered_df_copy['Timestamp'] = pd.to_datetime(pre_filtered_df_copy['Timestamp'])

        cab_age = pre_filtered_df_copy.groupby('DeviceName')['Timestamp'].apply(
            lambda date: (date.max() - date.min()).days).reset_index().rename(columns={'Timestamp': 'cab_age'})

        self.agg_table = pd.merge(agg_table, cab_age, how='left', on=['DeviceName'])

    def scale(self):
        numerical_columns = self.agg_table.select_dtypes([int, float]).columns
        self.agg_table_scaled = self.agg_table.copy()

        mm_scaler = MinMaxScaler()

        for col in numerical_columns:
            mm_scaler.fit(self.agg_table[[col]])
            self.agg_table_scaled[col] = mm_scaler.transform(self.agg_table_scaled[[col]])

    def one_hot_label_encode(self):
        self.encoded_df = self.agg_table_scaled.copy()
        one_hot_encoded_cols = ['Category', 'NewLocation', 'Segment_1']
        object_cols = self.encoded_df.select_dtypes('object').columns
        label_encoded_cols = set(object_cols) - set(one_hot_encoded_cols)
        # One hot encode
        self.encoded_df = pd.get_dummies(self.encoded_df, columns=one_hot_encoded_cols)
        # Label encode
        l_encoder = LabelEncoder()
        for feature in label_encoded_cols:
            l_encoder.fit(self.encoded_df[feature])
            self.encoded_df[feature] = l_encoder.transform(self.encoded_df[feature])

    @staticmethod
    def get_cluster_prediction(df, model):

        model.fit(df)
        prediction = model.predict(df)

        return prediction

    def remove_overlapping_clusters(self, df, cluster_col):
        cluster_map = {}
        for i in self.agg_table.DeviceName.unique():
            first_cluster = df[df.DeviceName == i][cluster_col].value_counts().reset_index().sort_values(cluster_col,
                                                                                                         ascending=False)[
                'index'][0]
            cluster_map[i] = first_cluster
        df[cluster_col] = df['DeviceName'].map(cluster_map)
        return cluster_map

    def run_steps(self):
        self.generate_aggregated_table()
        self.scale()
        self.one_hot_label_encode()
        # TODO: EVALUATE BEST METHOD
        # Observe that we only select 3 centroids
        # Observe that the fact that there are overlapping clusters
        em = GaussianMixture(n_components=3)
        prediction_em = self.get_cluster_prediction(df=self.encoded_df, model=em)
        self.agg_table['prediction_em'] = prediction_em
        cluster_map = self.remove_overlapping_clusters(df=self.agg_table, cluster_col='prediction_em')
        return cluster_map


class ModellingPipe:
    def __init__(self, pre_filtered_df1,
                 n_pct_cabinets=0.70,
                 cluster_map=None,
                 product_col_name='NameClean',
                 user_col_name='CustomerId',
                 percentile_thresh=0.65,
                 n_pct_barcodes=0.35,  # Used based pareto rule , it should be 20%
                 n_recommendations=10,
                 patience=5,
                 similarity_distance = 3
                 ):
        """
        Initializes an instance of the ModellingPipe class.

        Args:
            pre_filtered_df1: The pre-filtered dataframe.
            n_pct_cabinets: The percentage of cabinets to consider (default: 0.70).
            cluster_map: The clustering map (default: None).
            product_col_name: The column name for the product (default: 'NameClean').
            user_col_name: The column name for the user (default: 'CustomerId').
            percentile_thresh: The percentile threshold (default: 0.65).
            n_pct_barcodes: The percentage of barcodes to consider (default: 0.35).
            n_recommendations: The number of recommendations to generate (default: 10).
            patience: The patience value (default: 5).
            similarity_distance: The similarity distance (default: 3).
        """


        self.filtered_cab = None
        self.cluster_base = None
        self.base_matrix = None
        self.bundle_merged_df = None
        self.cluster_n = None
        self.n_pct_cabinets = n_pct_cabinets
        self.percentile_thresh = percentile_thresh
        self.n_pct_barcodes = n_pct_barcodes
        self.n_recommendations = n_recommendations
        self.patience = patience
        self.similarity_distance = similarity_distance
        self.product_col_name = product_col_name
        self.user_col_name = user_col_name

        # PREPROCESSING
        self.pre_filtered_df1 = pre_filtered_df1

        # CLUSTERING MAP
        self.cluster_map = cluster_map


        if self.cluster_map is not None:
            self.pre_filtered_df1['cabinet_cluster'] = self.pre_filtered_df1['DeviceName'].map(self.cluster_map)

    def define_cluster_n(self, cluster_n=1):

        # SELECT CLUSTER AND ADD ENGAGEMENT METRIC
        if self.cluster_map is not None:
            self.cluster_base = self.pre_filtered_df1[self.pre_filtered_df1.cabinet_cluster == cluster_n]
            self.cluster_n = self.cluster_base.copy()
        else:
            self.cluster_base = self.pre_filtered_df1.copy()
            self.cluster_n = self.cluster_base

        # THIS STEP SHOULD BE SEPARATED FROM THE SCRIPT ABOVE
        self.cluster_n['engagement_metric'] = self.cluster_base[self.product_col_name].apply(lambda x: 1 if x else 0)

    def get_bundle_order_ids(self):
        # GET BUNDLE IDS PER CLUSTER
        bundles_df = self.cluster_n.groupby('OrderId')[self.product_col_name].unique().reset_index()
        bundles_df['length'] = bundles_df[self.product_col_name].apply(lambda x: len(x))
        bundles_lower5 = bundles_df[(bundles_df['length'] > 1) & (bundles_df['length'] <= 5)]
        bundles_all = bundles_df[(bundles_df['length'] > 1)]

        def get_bundle_ids(bundle_mask):
            bundle_mask['ProductSet'] = bundle_mask[self.product_col_name].apply(lambda x: set(x))
            bundle_mask['normalized'] = bundle_mask['ProductSet'].apply(lambda x: ','.join([str(elem) for elem in x]))
            bundle_mask = set(bundle_mask['OrderId'].unique())
            return bundle_mask

        bundle_ids_lower5 = get_bundle_ids(bundles_lower5)
        bundle_ids_all = get_bundle_ids(bundles_all)
        return bundle_ids_lower5, bundle_ids_all

    @staticmethod
    def add_bundle_string(series):
        values = series.unique()
        if len(values) == 1:
            return values[0]
        else:
            return f"{values.tolist()[:]} bundle"

    def get_bundle_merged_dataframe(self):
        bundle_ids_lower5, bundle_ids_all = self.get_bundle_order_ids()
        bundle_t = self.cluster_n[self.cluster_n.OrderId.isin(bundle_ids_lower5)].groupby(
            ['CustomerId', 'OrderId']).agg(
            {self.product_col_name: self.add_bundle_string, 'engagement_metric': max})

        non_bundle_t = self.cluster_n[~self.cluster_n.OrderId.isin(bundle_ids_all)].groupby(
            ['CustomerId', self.product_col_name]).agg(
            {'engagement_metric': max})

        # Here is the group by already performed
        self.bundle_merged_df = pd.concat([non_bundle_t.reset_index(), bundle_t.reset_index()])
        self.bundle_merged_df.drop('OrderId', axis=1, inplace=True)

    def generate_matrix(self):
        # Observe that at this point we use the bundle merged dataframe
        print(f'Generating matrices with percentile_thresh: {self.percentile_thresh} ')
        p_matrix, self.base_matrix = h.return_csr_matrix(df=self.bundle_merged_df,
                                                         product_col_name=self.product_col_name,
                                                         user_col_name=self.user_col_name,
                                                         percentile_thresh=self.percentile_thresh
                                                         )
        # Calculate the sparsity level
        print(f'Computing sparsity level')
        num_zeros = self.base_matrix.size - np.count_nonzero(self.base_matrix.values)
        sparsity_level = num_zeros / self.base_matrix.size

        # Print the sparsity level
        print('Sparsity level:', sparsity_level)

    @staticmethod
    def from_deque_append_to_list(input_list: deque, container_list: list):

        for i in input_list:
            if i not in container_list:
                container_list.append(i)

    @staticmethod
    def flatten_bundles(top_similar_products, all_products):
        new_l = []
        for product in top_similar_products:
            if 'bundle' in product:
                product_edited = product.replace('bundle', '')
                sub_l = eval(product_edited)
                if all(elem in all_products for elem in sub_l):
                    pass
                else:
                    new_l.append(product)
            else:
                new_l.append(product)
        # Flatten the nested list (if any) to get a single list of strings
        flat_l = [item for sublist in new_l for item in sublist] if any(
            isinstance(i, list) for i in new_l) else new_l

        return flat_l

    @staticmethod
    def aggregate_best_worst_products(cabs_top_products: dict):
        dfs = list()
        for device in cabs_top_products.keys():
            t = pd.DataFrame.from_dict(cabs_top_products[device][0], orient='index').reset_index()
            t['products_to_remove'] = list(cabs_top_products[device][1])[:len(t)]
            t['DeviceName'] = device
            t.rename(columns={'index': 'top_products', 0: 'weighted_amount_products'}, inplace=True)
            dfs.append(t)
        return pd.concat(dfs)

    @staticmethod
    def generate_recommendations_df(device_name, final_recommendation):
        recommendations_df = pd.DataFrame(index=range(len(set(final_recommendation))))
        recommendations_df['DeviceName'] = device_name
        # Here we use a list to avoid error : 'set' type is unordered
        recommendations_df['top_n_recommendations'] = final_recommendation
        return recommendations_df

    @staticmethod
    def forward_operation(iterator):
        iterator += 1
        incremental_factor = 2
        patience = 0
        return iterator, incremental_factor, patience

    # TODO: REFACTOR THIS FUNCTION , BREAK THIS DOWN IN STEPS
    def generate_top_n_recommendations_by_top_products(self, content_agg_table, cosine_sim, device_name,
                                                       top_products_names):

        top_n_recommendations = list()
        # Loop trough products and get the TOP5 recommendations per product
        print(f'N products to be processed for device: {device_name}, {len(top_products_names)}')
        iterator = 0
        incremental_factor = 2
        patience_counter = 0

        while iterator < len(top_products_names):

            product_name = list(top_products_names.keys())[iterator]
            # Observe that function remove_similar_products_from_top_products removes all products that are not in matrix
            # so at this point all products stored in top_products_names are in matrix

            n_products = int(top_products_names[product_name])
            print(f'******* PRODUCT NAME: {product_name} *******')
            # Observe that we choose 10 for each recommender in order to have many initial products
            # Observe that incremental_factor increases the number of recommendations to ensure that we have unique
            # recommendations
            n_products_exp = int(n_products * incremental_factor)

            jaccard_similar_products = models.jaccard_item_based_recommender(product_id=product_name,
                                                                             sales_data=self.base_matrix,
                                                                             k=n_products_exp)

            content_similar_products = models.get_content_based_recs(content_agg_table=content_agg_table,
                                                                     cosine_sim=cosine_sim,
                                                                     product_name=product_name,
                                                                     n=n_products_exp)
            # Observe that we return n_products  from the hybrid recommender
            similar_products = models.hybrid_recsys_compiler(
                items_based_results=jaccard_similar_products[f'top{n_products_exp}_recommended_products'],
                content_based_results=content_similar_products[self.product_col_name],
                k=n_products_exp)
            print(
                f'1. PRODUCT {product_name}, SIMILAR PRODUCTS FOR PRODUCTS {similar_products} with n_products_exp {n_products_exp} ')

            final_recommendation = [i for i in similar_products if i not in
                                    set(self.filtered_cab[self.product_col_name])]

            print(f'2. PRODUCT {product_name},  FINAL RECOMMENDATION 1 {final_recommendation} ')

            if len(final_recommendation) >= n_products and self.check_recommendations_duplicates(
                    recommendations_container=top_n_recommendations,
                    final_recommendation=final_recommendation, n_recs=n_products):
                final_recommendation = final_recommendation[:n_products]
                print(f'3. PRODUCT {product_name},  FINAL RECOMMENDATION 2 {final_recommendation} ')

                iterator, incremental_factor, patience_counter = self.forward_operation(iterator)

                # Observe that at this point some products will be removed since these are duplicates
                self.from_deque_append_to_list(final_recommendation, top_n_recommendations)
                print(f'4. PRODUCT {product_name},  UPDATED LIST, FINAL RECOMMENDATION 3  {top_n_recommendations} ')

            else:
                patience_counter += 1
                if patience_counter > self.patience:
                    iterator, incremental_factor, patience_counter = self.forward_operation(iterator)
                else:
                    incremental_factor += 2

        return top_n_recommendations

    def generate_top_tail_products(self, filtered_cab, n_unique_products):

        n_products_based_pct_threshold = int(self.n_pct_barcodes * n_unique_products)

        print('n_products_based_pct_threshold', n_products_based_pct_threshold)
        tot_sum = filtered_cab.groupby(self.product_col_name)['totalSumEuro'].sum().sort_values(ascending=False)[
                  :n_products_based_pct_threshold].sum()
        weights = dict((filtered_cab.groupby(self.product_col_name)['totalSumEuro'].sum().sort_values(ascending=False)[
                        :n_products_based_pct_threshold] / tot_sum))
        n_products_needed = {k: np.round(weight * self.n_recommendations) for k, weight in weights.items()}

        if not sum(n_products_needed.values()) >= self.n_recommendations - 3:
            raise ValueError(f'Increase the number of n_recommendations, currently {self.n_recommendations}')

        tail_products_names = list(filtered_cab.groupby(self.product_col_name)[
                                       'totalSumEuro'].sum().sort_values(ascending=True)[
                                   :n_products_based_pct_threshold].keys())

        return tail_products_names, n_products_needed, weights

    #TODO: ADD COMMENTS WITH STEPS, REFACTOR
    def generate_output_compiler(self):

        # 1. Selecting top N cabinets from filtering
        dataframes = list()
        cabs_top_products = dict()

        # Observe that top_cabs are generated by selecting all the cabinets that are above the revenue mean
        # Observe that for Noury from a total of 66 cabinets we choose around 20 cabinets after the filter is applied

        top_cabs = md_helpers.get_top_cabs(self.cluster_base,self.n_pct_cabinets)

        content_agg_table, cosine_sim = models.get_agg_table_cosine_sim_matrix(cluster_base=self.cluster_base)
        for device_name in tqdm(list(top_cabs)):

            print('-----------------------------')
            print(f'DEVICE NAME: {device_name}')
            print('-----------------------------')

            # 2. Select only the latest days of the cabinet
            self.filtered_cab = h.get_cab_last_n_days(users_df_base=self.pre_filtered_df1, device_name=device_name)
            n_unique_products = self.filtered_cab[self.product_col_name].nunique()

            if n_unique_products >= 5:

                tail_products_names, n_products_needed, weights = self.generate_top_tail_products(
                    self.filtered_cab,
                    n_unique_products)


                n_products_needed = self.remove_similar_products_from_top_products(top_products_names=n_products_needed,
                                                                                   similarity_distance=self.similarity_distance)

                cabs_top_products[device_name] = (n_products_needed, set(tail_products_names))

                top_n_recommendations = self.generate_top_n_recommendations_by_top_products(content_agg_table,
                                                                                            cosine_sim,
                                                                                            device_name,
                                                                                            top_products_names=n_products_needed)

                # PRE SELECTION OF BUNDLES
                # Observe that this removes recommendations as bundles
                top_n_recommendations = self.flatten_bundles(top_n_recommendations,
                                                             set(self.filtered_cab[self.product_col_name]))

                # Include only cabinets with more than 3 recommendations
                if len(top_n_recommendations) >= 3:
                    recommendations_df = self.generate_recommendations_df(device_name, top_n_recommendations)
                    dataframes.append(recommendations_df)
                else:
                    print(f'Skipping cabinet due to lack in top_n_recommendations : {top_n_recommendations}')
            else:
                print(f'Skipping cabinet due to lack in n_unique_products : {device_name}')

        cabs_top_products = self.aggregate_best_worst_products(cabs_top_products)

        return pd.concat(dataframes), cabs_top_products

    # TODO: REMOVE COMMENTED BLOCKS, RENAME FUNCTION
    @staticmethod
    def check_recommendations_duplicates(recommendations_container, final_recommendation, n_recs):
        recommendations_container_set = set(recommendations_container)
        final_recommendation_set = set(final_recommendation)
        # print('recommendations_container_set', recommendations_container_set)
        # print('final_recommendation_set', final_recommendation_set)
        intersection = final_recommendation_set - recommendations_container_set
        # print('Intersection', intersection)
        # print(n_recs)
        l = list(intersection)[:n_recs]
        if len(l) == n_recs:
            return True

    @staticmethod
    def new_weights(p, dup, top_products_names):
        output = {}
        for key, value in top_products_names.items():
            if key == p or key not in dup:
                output[key] = value
        for key in dup:
            if key in top_products_names and key != p:
                output[p] = output.get(p, 0) + top_products_names[key]
        return output

    def remove_similar_products_from_top_products(self, top_products_names, similarity_distance=3):
        i = 0
        excluded_products = list()

        while i < len(top_products_names):
            product_name = list(top_products_names.keys())[i]
            if product_name in self.base_matrix.index:
                jaccard_similar_products = models.jaccard_item_based_recommender(product_id=product_name,
                                                                                 sales_data=self.base_matrix,
                                                                                 k=similarity_distance)
                jaccard_similar_products = list(
                    jaccard_similar_products[f'top{similarity_distance}_recommended_products'])
                # Update
                dup = [i for i in list(top_products_names.keys()) if i in jaccard_similar_products]
                top_products_names = self.new_weights(p=product_name, dup=dup, top_products_names=top_products_names)
            else:
                print(f'Product {product_name} is not in matrix, filter percentile_thresh has excluded it')
                excluded_products.append(product_name)

            i += 1

        top_products_names = {k: v for k, v in top_products_names.items() if k not in excluded_products}
        return top_products_names

    def run_steps(self):

        if self.cluster_map is None:
            self.define_cluster_n()

            # Observe that this step contains the add_bundle_string ids generation
            self.get_bundle_merged_dataframe()
            self.generate_matrix()

            final_output, cabs_top_products = self.generate_output_compiler()
            return final_output, cabs_top_products

        else:
            # Here we select the observed clusters
            aggregated_outputs = list()
            cabs_top_products_aggregated = list()
            for cluster in set(self.cluster_map.values()):
                self.define_cluster_n(cluster_n=cluster)
                # Observe that this step contains the add_bundle_string ids generation
                self.get_bundle_merged_dataframe()
                self.generate_matrix()
                final_output, cabs_top_products = self.generate_output_compiler()
                aggregated_outputs.append(final_output)
                cabs_top_products_aggregated.append(cabs_top_products)

            return pd.concat(aggregated_outputs).reset_index(drop=True), pd.concat(
                cabs_top_products_aggregated).reset_index(drop=True)


