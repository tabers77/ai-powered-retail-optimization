def df_has_duplicate_rows(df):
    # Observe that this function will fail with list format
    if len(df[df.duplicated()]) > 0:
        return True
    else:
        return False

