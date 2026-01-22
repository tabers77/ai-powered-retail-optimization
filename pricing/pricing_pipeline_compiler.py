import md_helpers
from pricing.pricing_model_factory import ThomsonSampler
import pandas as pd
from global_preprocessor import GlobalPreprocessor

class Compilers:
    def __init__(self,df, product_col_name= 'name', price_col_name= 'price', orgs_ids_list=None):
        self.df = df
        self.product_col_name = product_col_name
        self.price_col_name = price_col_name
        self.copy_df = None
        self.top_cabs = None
        self.orgs_ids_list = orgs_ids_list
    def preprocessing(self, n_pct_cabinets, n_days_limit, start_date='2022-01-01'):
        print('Running preprocessing...')
        # -----------------------------------------------
        # TEST
        glob_preprocessor = GlobalPreprocessor(df=self.df, start_date=start_date, orgs_ids_list=self.orgs_ids_list)
        self.copy_df = glob_preprocessor.run_preprocessor()
        # -----------------------------------------------

        #self.copy_df = md_helpers.preprocessor1_df_level(self.df, start_date=start_date)

        self.top_cabs = md_helpers.get_top_cabs_based_on_barcodes(self.copy_df,
                                                                  n_pct_cabinets=n_pct_cabinets,
                                                                  n_days_limit=n_days_limit

                                                                  )


    def train_compiler(self, n_iterations=150, prior_init_type='mean_based', visualize_samples=False):
        if self.copy_df is None and self.top_cabs is None :
            raise ValueError('Please run the preprocessing step first')

        print('Running modelling...')

        output_dataframes = list()
        for cab_name in self.top_cabs:

            cabinet_df = self.copy_df[self.copy_df.DeviceName == cab_name]
            mean_demand = cabinet_df.groupby([self.product_col_name ]).agg({'TotalCount': sum}).mean().item()
            # Select products with a demand higher than mean
            cabinet_df_agg = cabinet_df.groupby(self.product_col_name ).agg({'TotalCount': sum}).reset_index()
            products_list = cabinet_df_agg[cabinet_df_agg.TotalCount > mean_demand][self.product_col_name ].to_list()
            relevant_cabinets_df = cabinet_df[cabinet_df.name.isin(products_list)]

            # Select cabinet with multi price products
            mask = relevant_cabinets_df.groupby(self.product_col_name)[self.price_col_name].nunique() >= 2
            multi_price_products = list(
                relevant_cabinets_df[relevant_cabinets_df[self.product_col_name].isin(mask[mask].index)][
                    self.product_col_name].unique())

            product_prices_container = dict()

            for product_name in multi_price_products:
                print(f'Processing product: {product_name}')
                agg_by_price_df = relevant_cabinets_df[relevant_cabinets_df.name == product_name].groupby(self.price_col_name)['TotalCount'].sum().reset_index()
                sampler = ThomsonSampler(prices_to_test=agg_by_price_df.price, demands=agg_by_price_df['TotalCount'])
                opt_price_probs = sampler.baseline2(n_iterations=n_iterations, prior_init_type=prior_init_type,
                                                    visualize_samples=visualize_samples)
                product_prices_container[product_name] = opt_price_probs
                # ADD DEMANDS AT THIS POINT

            cab_output_df = pd.DataFrame.from_dict(product_prices_container, orient='index').stack().reset_index()
            cab_output_df.columns = [self.product_col_name , self.price_col_name, 'optimal_price_prob']
            cab_output_df['DeviceName'] = cab_name
            last_column = cab_output_df.pop('DeviceName')
            cab_output_df.insert(0, 'DeviceName', last_column)

            output_dataframes.append(cab_output_df)

        return pd.concat(output_dataframes)