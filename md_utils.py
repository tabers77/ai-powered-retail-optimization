import numpy as np
import pandas as pd
from sklearn.preprocessing import FunctionTransformer


def sin_transformer(period):
    return FunctionTransformer(lambda x: np.sin(x / period * 2 * np.pi))


def cos_transformer(period):
    return FunctionTransformer(lambda x: np.cos(x / period * 2 * np.pi))


def apply_cyclical_encoding(unique_barcode_df):
    """
    Apply cyclical one_hot_label_encode to the month, day, and hour features of the input dataframe.

    Parameters:
    unique_barcode_df (DataFrame): The input dataframe with month, day, and hour features.

    Returns:
    DataFrame: The input dataframe with additional sin and cos features for the month, day, and hour features.
    """
    # Select the features we need in order to perform one_hot_label_encode
    copy_df = unique_barcode_df.copy()
    copy_df = copy_df[['Month', 'Day', 'Hour']]
    # Month
    copy_df["month_sin"] = sin_transformer(12).fit_transform(copy_df)["Month"]
    copy_df["month_cos"] = cos_transformer(12).fit_transform(copy_df)["Month"]
    # Day
    copy_df["day_sin"] = sin_transformer(365).fit_transform(copy_df)["Day"]
    copy_df["day_cos"] = cos_transformer(365).fit_transform(copy_df)["Day"]
    # Hour
    copy_df["hour_sin"] = sin_transformer(8760).fit_transform(copy_df)["Hour"]
    copy_df["hour_cos"] = cos_transformer(8760).fit_transform(copy_df)["Hour"]

    output = copy_df[['month_sin', 'month_cos', 'day_sin', 'day_cos', 'hour_sin', 'hour_cos']]
    # Observe that we need to perform a concatenation operation at this point
    return pd.concat([unique_barcode_df, output], axis=1)


def get_mapes_confidence_label(x):
    if x <= 0.30:
        return 'high_confidence'
    elif 0.30 < x <= 0.70:
        return 'medium_confidence'
    else:
        return 'low_confidence'

