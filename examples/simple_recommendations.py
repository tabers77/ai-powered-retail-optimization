"""
Simple Recommender System Example

This script demonstrates basic usage of the recommendation pipeline.
"""

import pandas as pd
from recommenders.recsys_pipeline_compiler import Preprocessing, Clustering, ModellingPipe

def main():
    # Configuration
    DATA_PATH = 'data/transactions.csv'
    N_BARCODES_PCT = 0.30  # Top 30% of products per cabinet
    PERCENTILE_THRESH = 0.80  # Use 80th percentile threshold
    N_PCT = 0.20  # Top 20% recommendation threshold
    START_DATE = '2021-06-01'
    
    print("Loading transaction data...")
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df)} transactions")
    
    # Step 1: Preprocessing
    print("\n[Step 1/3] Preprocessing data...")
    preprocessor = Preprocessing(
        df=df,
        n_barcodes_pct=N_BARCODES_PCT,
        start_date=START_DATE
    )
    pre_filtered_df1, pre_filtered_df2 = preprocessor.run_steps()
    print(f"Filtered to {pre_filtered_df2['DeviceName'].nunique()} cabinets")
    print(f"Working with {pre_filtered_df2['barcode'].nunique()} products")
    
    # Step 2: Clustering
    print("\n[Step 2/3] Clustering similar cabinets...")
    clusterer = Clustering(pre_filtered_df2)
    cluster_map = clusterer.run_steps()
    print(f"Created {len(set(cluster_map.values()))} clusters")
    
    # Step 3: Generate Recommendations
    print("\n[Step 3/3] Generating recommendations...")
    model_object = ModellingPipe(
        pre_filtered_df1=pre_filtered_df1,
        cluster_map=cluster_map,
        product_col_name='NameClean',
        user_col_name='CustomerId',
        percentile_thresh=PERCENTILE_THRESH,
        n_pct=N_PCT
    )
    
    recommendations, top_products = model_object.run_steps()
    
    print("\n✓ Recommendation generation complete!")
    print(f"Generated recommendations for {len(recommendations)} cabinet-product combinations")
    
    # Save results
    output_dir = 'outputs'
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    recommendations.to_csv(f'{output_dir}/recommendations.csv', index=False)
    top_products.to_csv(f'{output_dir}/top_products_per_cabinet.csv', index=False)
    
    print(f"\nResults saved to {output_dir}/")
    
    # Display sample recommendations
    print("\nSample Recommendations:")
    print(recommendations[['DeviceName', 'NameClean', 'recommendation_score']].head(15))
    
    # Display cabinet-level summary
    if 'DeviceName' in top_products.columns:
        print("\nTop Products per Cabinet:")
        for cabinet in top_products['DeviceName'].unique()[:3]:
            cabinet_products = top_products[top_products['DeviceName'] == cabinet]
            print(f"\n{cabinet}:")
            print(cabinet_products[['NameClean', 'score']].head(5).to_string(index=False))

if __name__ == "__main__":
    main()
