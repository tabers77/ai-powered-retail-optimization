import pandas as pd


class GlobalPreprocessor:
    def __init__(self, df, start_date ='2021-06-01', orgs_ids_list=None):
        self.df1 = None
        self.orgs_ids_list = orgs_ids_list
        self.df = df.copy()
        self.start_date = start_date

    # TODO: ADD PRINT STATEMENTS
    def s1_fix_device_names(self):
        """
        Fix device name inconsistencies in the dataset.
        This handles cases where the same physical cabinet had different device IDs over time.
        """
        # ----------------
        # Cabinet Location A
        # ----------------
        self.df1 = self.df.copy()

        mask1 =  self.df1[
            ( self.df1.DeviceId == 'if-11-03-2021-e87f') & ((self.df1.Date >= '2021-04-30') & (self.df1.Date <= '2023-01-27'))].index
        self.df1.loc[mask1, 'DeviceName'] = '3B6N | Cabinet_Location_A'

        mask2 = self.df1[
            (self.df1.DeviceId == 'if-23-6-2022-818172') & ((self.df1.Date >= '2023-01-27') & (self.df1.Date <= '2023-01-31'))].index
        self.df1.loc[mask2, 'DeviceName'] = '3B6N | Cabinet_Location_A'

        # ----------------
        # Cabinet Location B
        # ----------------
        mask3 = self.df1[
            (self.df1.DeviceId == 'if-11-03-2021-079b') & ((self.df1.Date >= '2021-04-30') & (self.df1.Date <= '2022-11-08'))].index
        self.df1.loc[mask3, 'DeviceName'] = '3B6A | Cabinet_Location_B'

        return self.df1

    def s2_preprocessor1_df_level(self):
        """
        Preprocesses the input dataframe by converting the 'Timestamp' and 'Date' columns to datetime objects, and adding
        a 'week' column representing the week of the year for the 'Timestamp' column.

        Parameters:
        df (pd.DataFrame): input dataframe containing the columns 'Timestamp' and 'Date'.

        Returns:
        pd.DataFrame: preprocessed dataframe with additional 'week' column.
        """
        if self.df1 is not None:
            if self.orgs_ids_list is not None:
                self.df1 = self.df1[self.df1.Organization_Id.isin(self.orgs_ids_list)]

            self.df1 = self.df1[self.df1.Date >= self.start_date]
            self.df1['Timestamp'] = pd.to_datetime(self.df1['Timestamp'])
            self.df1['week'] = self.df1['Timestamp'].dt.strftime('%W')
            self.df1['Date'] = pd.to_datetime(self.df1['Date'])
            self.df1['DeviceName'] = self.df1['DeviceName'] + '-' + self.df1['NewLocation']

            return self.df1
        else:
            raise ValueError(f'Please call the function {self.s1_fix_device_names.__name__} first')



    def run_preprocessor(self):
        self.s1_fix_device_names()
        df = self.s2_preprocessor1_df_level()
        return df
