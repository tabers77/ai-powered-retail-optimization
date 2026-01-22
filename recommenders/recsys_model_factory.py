import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import deque
import numpy as np
from sklearn.neighbors import NearestNeighbors


def get_item_based_recommendations(product_matrix, matrix, n_recommendations=10):
    """
      This uses K-nearest neighbors
      """

    model_knn = NearestNeighbors(metric='cosine', algorithm='brute')
    model_knn.fit(product_matrix)
    model_knn.fit(product_matrix)

    dataframes = []

    # Here we loop through all the products in the matrix
    for product in matrix.index:

        product_index = matrix.index
        product_index = int(product_index.get_loc(product))

        distances, indices = model_knn.kneighbors(matrix.iloc[product_index, :].values.reshape(1, -1),
                                                  n_neighbors=n_recommendations + 1)

        container = {}
        for i in range(1, len(distances.flatten())):
            key = matrix.index[indices.flatten()[i]]
            value = distances.flatten()[i]
            container[key] = value

        df = pd.DataFrame(container.items()).rename(
            columns={0: f'top{n_recommendations}_recommended_products', 1: 'distance'})
        df['product_name'] = product
        df['rank'] = df.distance.rank(ascending=True)
        df = df[['product_name', f'top{n_recommendations}_recommended_products', 'distance', 'rank']]
        dataframes.append(df)

    output_df = pd.concat(dataframes).reset_index(drop=True)

    return output_df


def get_tfidf_matrix(df, cols):
    copy_df = df.copy()
    tfidf = TfidfVectorizer(stop_words='english')
    for col in cols:
        copy_df[col] = copy_df[col].fillna('')

    tfidf_matrix = tfidf.fit_transform(
        copy_df['Category'] + ' ' + copy_df['NewLocation'] + ' ' + copy_df['locationName'] + ' ' +
        copy_df['Segment_1'] + ' ' + copy_df['ProductTypeClean'] + ' ' +
        copy_df['price_label'] + ' ' + copy_df['orders_frequency_label'])

    return tfidf_matrix


def get_agg_table_cosine_sim_matrix(cluster_base):
    # TODO: SPLIT THESE STEPS

    # 1. GENERATING LABELS
    df_with_labels = cluster_base.groupby(['NameClean']).agg({'priceEuro': 'mean', 'OrderId': 'nunique'}).reset_index()
    df_with_labels.rename(columns={'OrderId': 'orders_frequency'}, inplace=True)
    df_with_labels['price_label'] = pd.qcut(df_with_labels['priceEuro'], q=3, labels=['low', 'medium', 'high'])
    df_with_labels['price_label'] = df_with_labels['price_label'].astype(str)
    df_with_labels['orders_frequency_label'] = pd.qcut(df_with_labels['orders_frequency'], q=3, labels=['low', 'medium', 'high'])
    df_with_labels['orders_frequency_label'] = df_with_labels['orders_frequency_label'].astype(str)

    # 2. AGGREGATION BASED ON OBSERVED COLUMNS
    content_agg_table = cluster_base.groupby(
        ['NameClean', 'Category', 'NewLocation', 'locationName', 'Segment_1', 'ProductTypeClean']).agg(
        {'priceEuro': 'mean'}).reset_index()
    content_agg_table = pd.merge(content_agg_table, df_with_labels[['NameClean', 'price_label', 'orders_frequency_label']],
                                 how='left',
                                 on='NameClean')
    # 3. GENERATING THE MATRIX
    tfidf_matrix = get_tfidf_matrix(content_agg_table,
                                    ['Category', 'NewLocation', 'locationName', 'Segment_1', 'price_label',
                                     'orders_frequency_label'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return content_agg_table, cosine_sim


def get_content_based_recs(content_agg_table, cosine_sim, product_name, n=5):
    # Get the index of the product in the dataframe
    product_index = content_agg_table[content_agg_table['NameClean'] == product_name].index[0]
    # Get the cosine similarity scores between the product and all others products
    sim_scores = list(enumerate(cosine_sim[product_index]))
    # Sort the scores in descending order
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    # Get the indices of the top n similar products
    top_indices = [i[0] for i in sim_scores[1:n + 1]]
    # Return the product names and other features of the top n similar products
    return content_agg_table.iloc[top_indices]


def hybrid_recsys_compiler(items_based_results: pd.Series, content_based_results: pd.Series, k=10):
    """
    recsys1_result expects a pandas series like df['NameClean']
    :param content_based_results:
    :param items_based_results:
    :param k:
    :return:
    """

    hybrid_recommendations = deque()

    intersection = set(items_based_results) & set(content_based_results)
    n = k - len(intersection)
    # Initially we could assign a higher weight to item based recommender and not content based
    n_items_from_recsys1 = round(n * 0.6)
    n_items_from_recsys2 = round(n * 0.4)

    for i in intersection:
        hybrid_recommendations.append(i)

    # Remove duplicates
    cleaned_list_item_based = []
    [cleaned_list_item_based.append(i) for i in items_based_results if i not in intersection and i not in cleaned_list_item_based]

    cleaned_list_content_based = []
    [cleaned_list_content_based.append(i) for i in content_based_results if i not in intersection and i not in cleaned_list_content_based]

    cleaned_list_item_based = cleaned_list_item_based[:n_items_from_recsys1]
    cleaned_list_content_based = cleaned_list_content_based[:n_items_from_recsys2]

    hybrid_recommendations.extend(cleaned_list_item_based)
    hybrid_recommendations.extend(cleaned_list_content_based)

    return hybrid_recommendations


# Define a function to calculate the Jaccard similarity between two products
def jaccard_similarity(product1, product2):
    intersection = np.sum(np.logical_and(product1, product2))
    union = np.sum(np.logical_or(product1, product2))
    return intersection / union


# Define a function to recommend similar products
def jaccard_item_based_recommender(product_id, sales_data, k=10):
    # Calculate the Jaccard similarity between the target product and all others products
    similarities = {p: jaccard_similarity(sales_data.loc[product_id], sales_data.loc[p]) for p in sales_data.index if
                    p != product_id}
    t = pd.DataFrame.from_dict(similarities.items())
    t.rename(columns={0: f'top{k}_recommended_products', 1: f'similarity_metric'}, inplace=True)
    t = t[t[f'similarity_metric'] > 0].sort_values(f'similarity_metric', ascending=False)[
        :k].reset_index(drop=True)
    return t
