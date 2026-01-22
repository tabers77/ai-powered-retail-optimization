"""
Simple Forecasting Example

This script demonstrates basic usage of the forecasting pipeline.
Run this after preparing your data in the data/ directory.
"""

import pandas as pd
from forecasting.forecasting_pipeline_compiler import Compilers

def main():
    # Configuration
    DATA_PATH = 'data/transactions.csv'
    START_DATE = '2021-06-01'
    N_PCT_CABINETS = 0.20  # Use top 20% of cabinets
    N_DAYS_LIMIT = 30       # Minimum 30 days of data required
    TRAIN_SIZE = 0.85       # 85% train, 15% test
    
    print("Loading transaction data...")
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df)} transactions")
    
    print(f"\nInitializing forecasting pipeline...")
    compiler = Compilers(
        df=df,
        n_days_limit=N_DAYS_LIMIT,
        n_pct_cabinets=N_PCT_CABINETS,
        train_size=TRAIN_SIZE,
        sparsity_level=0.75,
        infer_mode=False,
        start_date=START_DATE
    )
    
    print("Running forecasting pipeline...")
    print("This may take several minutes...")
    
    # Run the complete pipeline
    # This will train models and generate predictions
    predictions = compiler.run_all_steps()
    
    print(f"\nForecasting complete!")
    print(f"Generated predictions for {predictions['DeviceName'].nunique()} cabinets")
    print(f"Total predictions: {len(predictions)}")
    
    # Save results
    output_file = 'outputs/forecasting_results.csv'
    predictions.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")
    
    # Display sample results
    print("\nSample predictions:")
    print(predictions[['datetime', 'DeviceName', 'barcode', 'name', 'actuals', 'predictions']].head(10))
    
    # Display performance metrics
    if 'mean_squared_error' in predictions.columns:
        print("\nPerformance Summary:")
        print(f"Average MSE: {predictions['mean_squared_error'].mean():.2f}")
        print(f"Average RMSE: {predictions['root_mean_squared_error'].mean():.2f}")
        print(f"Average MAPE: {predictions['mape'].mean():.2%}")

if __name__ == "__main__":
    # Create outputs directory if it doesn't exist
    import os
    os.makedirs('outputs', exist_ok=True)
    
    main()
