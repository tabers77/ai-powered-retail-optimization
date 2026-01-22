from scipy.sparse import csr_matrix
import pandas as pd


def generate_base_matrix(dataframe, product_col_name, user_col_name, percentile_thresh=0.65):
    """
    Generates the initial matrix
    """

    # Count based on interactions
    product_count = dataframe.groupby(by=[product_col_name])['engagement_metric'].count().reset_index().rename(
        columns={'engagement_metric': 'interactions_count'})
    # Generate count of the ratings
    total_rating_count = dataframe.merge(product_count, left_on=product_col_name, right_on=product_col_name,
                                         how='left')
    print(f'Number of unique products in total {total_rating_count[product_col_name].nunique()}')

    # Choose data frame containing only popular products' ion order to reduce sparsity  -  Here I filter out non-
    # popular products
    popularity_threshold_quant = int(
        total_rating_count.groupby(product_col_name)['interactions_count'].max().quantile(percentile_thresh))

    interactions_popular_product = total_rating_count[
        total_rating_count['interactions_count'] >= popularity_threshold_quant]

    print(
        f'Number of unique products after filtered {interactions_popular_product[product_col_name].nunique()}')

    nan_matrix = pd.pivot_table(interactions_popular_product, values='engagement_metric',
                                index=product_col_name,
                                columns=user_col_name)
    # Fill Nan values
    trans1_matrix = nan_matrix.fillna(0)
    trans2_matrix = nan_matrix.apply(lambda row: row.fillna(row.mean()), axis=1)

    # Observe that this operation is too expensive
    #trans3_matrix = nan_matrix[nan_matrix[:] > 0].fillna(nan_matrix.mean(axis=0))

    return nan_matrix, trans1_matrix, trans2_matrix #, trans3_matrix


def return_csr_matrix(df, product_col_name, user_col_name, percentile_thresh=0.65):
    tup = generate_base_matrix(df, product_col_name, user_col_name, percentile_thresh=percentile_thresh)
    matrix = tup[1]

    return csr_matrix(matrix.values), matrix


def get_cabs_above_revenue_mean(cluster_base, start_date='2022-01-01'):
    cluster_base = cluster_base[cluster_base.Timestamp >= start_date]

    mean_revenue = cluster_base.groupby('DeviceName')['totalSumEuro'].sum().mean()
    cabs = cluster_base.groupby('DeviceName')['totalSumEuro'].sum().reset_index()
    top_cabs = set(cabs[cabs['totalSumEuro'] >= mean_revenue]['DeviceName'])

    return top_cabs


def get_cab_last_n_days(users_df_base, device_name):
    thresh = 0.30

    cab = users_df_base[users_df_base.DeviceName == device_name]

    # Choose days for start_date (Ex: last 3 months) to handle discontinued products
    days_cab = (pd.to_datetime(cab['Timestamp'].max()) - pd.to_datetime(cab['Timestamp'].min())).days
    days = int(thresh * days_cab)
    start_date = pd.to_datetime(cab['Timestamp'].max()) - pd.Timedelta(days=days)
    output_cab = cab[pd.to_datetime(cab['Timestamp']) >= start_date]
    return output_cab
