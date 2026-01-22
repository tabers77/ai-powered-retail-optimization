# AI-Powered Retail Optimization: Time Series Forecasting & Recommender Systems

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Machine Learning](https://img.shields.io/badge/ML-Forecasting%20%7C%20RecSys-green.svg)](https://github.com)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **Portfolio Project**: Advanced machine learning solutions for retail inventory optimization, featuring time series forecasting and personalized product recommendations.

## 🎯 TL;DR (For Recruiters)

- **Built ML systems** to optimize retail inventory and product assortment
- **Forecasted cabinet-level demand** using LSTM + attention (RMSE ↓ to 10.9)
- **Designed hybrid recommender system** (collaborative + content-based)
- **Reduced waste and improved product matching** through data-driven decisions
- **Tech Stack**: Python, LSTM, XGBoost, Recommender Systems, MLflow

## 👨‍💻 My Role

- Designed end-to-end forecasting and recommender pipelines
- Engineered temporal, weather, and holiday features
- Built and evaluated LSTM with attention mechanism
- Designed hybrid recommender system architecture
- Conducted hyperparameter optimization and experiment tracking

---

## 📋 Table of Contents
- [TL;DR](#-tldr-for-recruiters)
- [My Role](#-my-role)
- [Overview](#-overview)
- [Project Components](#-project-components)
- [Final Model Selection](#-final-model-selection)
- [Results Snapshot](#-results-snapshot)
- [Technical Stack](#-technical-stack)
- [Model Architectures](#-model-architectures)
- [Experimental Insights](#-experimental-insights)
- [Installation](#-installation)
- [Usage Examples](#-usage-examples)
- [Future Enhancements](#-future-enhancements)
- [About](#-about)

---

## 🎯 Overview

This repository showcases two interconnected machine learning systems designed to optimize retail operations by reducing waste and improving customer satisfaction:

1. **Time Series Forecasting Model**: Predicts weekly sales at the individual cabinet level to optimize inventory management
2. **Recommender System**: Provides personalized product recommendations using hybrid collaborative and content-based filtering

### Business Problem

Retail smart cabinets face two critical challenges:
- **Excess Waste**: Over-stocking leads to product expiration and waste
- **Suboptimal Product Selection**: Lack of personalization reduces sales potential

### Solution Impact

- ✅ Accurate demand forecasting reduces waste by predicting exact quantities needed
- ✅ Personalized recommendations increase sales through better product-cabinet matching
- ✅ Data-driven inventory management improves operational efficiency

---

## 🔬 Project Components

### 1️⃣ Time Series Forecasting System

**Objective**: Predict the quantity of products that will be sold per cabinet to minimize waste and optimize inventory.

**Approach**:
- Ensemble methods (Random Forest, XGBoost)
- LSTM neural networks (multivariate & univariate with attention mechanism)
- Comprehensive feature engineering (seasonality, trends, weather, holidays)

**Key Innovations**:
- Multivariate LSTM with attention layers for improved accuracy
- Integration of external features (weather, holidays) 
- Daily aggregation with 14-day sliding window optimization
- Hybrid model comparison framework

### 2️⃣ Recommender System

**Objective**: Recommend optimal product assortments for each cabinet based on historical performance and product attributes.

**Approach**:
- Hybrid recommendation combining:
  - **Item-based Collaborative Filtering** using Jaccard similarity
  - **Content-based Filtering** using product attributes
- Clustering-based approach for scalable recommendations
- Top-performing product filtering

**Key Innovations**:
- Cabinet clustering based on location, product mix, and performance metrics
- Weighted hybrid approach (60% collaborative, 40% content-based)
- Matrix factorization with SVD for sparsity handling

---

## ✅ Final Model Selection

### Forecasting Model
**Selected: Multivariate LSTM with Attention**
- ✅ **Lowest RMSE**: 10.90 (vs 11.39 univariate, 11.87 XGBoost, 12.34 Random Forest)
- ✅ **Strong performance with external features**: Weather reduced error by 5-8%, holidays by 2-3%
- ✅ **Stable training**: Early stopping with patience=5, NAdam optimizer
- ✅ **Optimal hyperparameters**: 32 LSTM units, 14-day window, batch size 64, MSE loss
- ✅ **Training time**: ~43 minutes for 20% of cabinets (comparable to univariate LSTM)

**Why not ensemble methods?**
- Random Forest & XGBoost had higher RMSE (11.87-12.34)
- LSTM captured temporal dependencies better
- Attention mechanism improved multivariate performance by 8-12%

### Recommender System
**Selected: Hybrid (Collaborative + Content-Based) with Clustering**
- ✅ **Better diversity**: 25% more diverse than pure collaborative filtering
- ✅ **Scales efficiently**: Clustering reduced computation time by 60%
- ✅ **Balanced approach**: 60% collaborative (user behavior) + 40% content-based (product attributes)
- ✅ **Quality threshold**: Maintains 80th percentile for top products
- ✅ **Optimal clustering**: Gaussian Mixture Model with 3 components

**Why hybrid over pure collaborative?**
- Pure collaborative filtering struggled with sparse user-item matrix
- Content-based ensures product diversity and handles cold-start
- Jaccard similarity outperformed cosine for binary transaction data

---

## 📊 Results Snapshot

### Forecasting Performance

| Model | RMSE | MAE | Training Time* | Selected |
|-------|------|-----|----------------|----------|
| Random Forest | 12.34 | 8.92 | ~15 min | ❌ |
| XGBoost | 11.87 | 8.45 | ~18 min | ❌ |
| Univariate LSTM | 11.39 | 8.21 | ~43 min | ❌ |
| **Multivariate LSTM + Attention** | **10.90** | **7.85** | ~43 min | ✅ |

*Training time for 20% of cabinets on standard compute

### Recommender Performance

| Metric | Result |
|--------|--------|
| Computation Time Reduction | 60% (via clustering) |
| Recommendation Diversity | +25% vs pure collaborative |
| Quality Threshold | 80th percentile maintained |
| Cluster Count | 3 (optimal) |
| Weighting | 60% collaborative + 40% content |

### Key Impact Metrics
- ✅ **Feature Importance**: Historical sales > day of week > month > weather > holidays
- ✅ **Optimal Window**: 14 days for forecasting
- ✅ **Best Optimizer**: NAdam outperformed Adam and RMSprop
- ✅ **Daily > Hourly/Weekly**: Daily aggregation performed best

---

## 🛠 Technical Stack

**Core Technologies**:
- **Python 3.8+**: Primary programming language
- **Pandas & NumPy**: Data manipulation and numerical computing
- **Scikit-learn**: Machine learning algorithms and preprocessing
- **TensorFlow/Keras**: Deep learning (LSTM models)
- **XGBoost**: Gradient boosting

**Key Libraries**:
```
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=0.24.0
tensorflow>=2.6.0
xgboost>=1.4.0
matplotlib>=3.4.0
seaborn>=0.11.0
mlflow>=1.20.0
tqdm>=4.62.0
```

---

## 📁 Project Structure

```
retail-ai-optimization/
│
├── forecasting/                        # Time Series Forecasting Module
│   ├── forecasting_model_factory.py   # Model creation and management
│   ├── forecasting_pipeline_compiler.py # Training pipeline orchestration
│   ├── forecasting_preprocessors.py   # Data preprocessing specific to forecasting
│   └── modelling_pipelines.py         # Model training pipelines
│
├── recommenders/                       # Recommender System Module
│   ├── recsys_helpers.py              # Helper functions for recommender
│   ├── recsys_model_factory.py        # Recommendation model factory
│   └── recsys_pipeline_compiler.py    # Recommendation pipeline
│
├── pricing/                            # Pricing Module (Future Enhancement)
│   ├── pricing_model_factory.py
│   └── pricing_pipeline_compiler.py
│
├── models_tests/                       # Testing utilities
│   └── tests_utils.py
│
├── global_preprocessor.py              # Global data preprocessing
├── md_configs.py                       # Model configurations
├── md_constants.py                     # Constants and parameters
├── md_dataset_factory.py               # Dataset creation utilities
├── md_evaluators.py                    # Model evaluation functions
├── md_helpers.py                       # General helper functions
├── md_hyperopt.py                      # Hyperparameter optimization
├── md_utils.py                         # Utility functions
├── md_visualizers.py                   # Visualization tools
├── mlflow_logs.py                      # MLflow experiment tracking
├── scorers.py                          # Custom scoring functions
├── requirements.txt                    # Python dependencies
└── README.md                           # Project documentation
```

---
## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Virtual environment (recommended)

### Setup Instructions

1. **Clone the repository**:
```bash
git clone https://github.com/tabers77/retail-ai-optimization.git
cd retail-ai-optimization
```

2. **Create and activate virtual environment**:
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

4. **Prepare data directory**:
```bash
mkdir data
# Place your transaction CSV files in the data/ directory
```

---

## 💡 Usage Examples

### Time Series Forecasting

```python
from forecasting.forecasting_pipeline_compiler import Compilers
from global_preprocessor import GlobalPreprocessor
import pandas as pd

# Load transaction data
df = pd.read_csv('data/transactions.csv')

# Initialize and run forecasting pipeline
compiler = Compilers(
    df=df,
    n_days_limit=30,
    n_pct_cabinets=0.20,
    train_size=0.85,
    sparsity_level=0.75,
    start_date='2021-06-01'
)

# Generate predictions
predictions_df = compiler.run_forecasting_pipeline()
```

### Recommender System

```python
from recommenders.recsys_pipeline_compiler import Preprocessing, Clustering, ModellingPipe

# Step 1: Preprocessing
preprocessor = Preprocessing(df=df, n_barcodes_pct=0.30)
pre_filtered_df1, pre_filtered_df2 = preprocessor.run_steps()

# Step 2: Clustering
clusterer = Clustering(pre_filtered_df2)
cluster_map = clusterer.run_steps()

# Step 3: Generate recommendations
model_object = ModellingPipe(
    pre_filtered_df1=pre_filtered_df1,
    cluster_map=cluster_map,
    product_col_name='NameClean',
    user_col_name='CustomerId',
    percentile_thresh=0.80,
    n_pct=0.20
)

recommendations, top_products = model_object.run_steps()
```

---

## 🏗 Model Architectures

### Forecasting Architecture

The forecasting system uses a multi-stage pipeline:

1. **Data Preprocessing**
   - Date filtering and transformation
   - Device name standardization
   - Missing value handling
   - Feature extraction (cyclical encoding for temporal features)

2. **Feature Engineering**
   - **Temporal Features**: Year, month, week, day, weekday, weekend indicator
   - **Cyclical Encoding**: Sin/cos transformations for seasonality
   - **External Features**: Weather (temperature, humidity), holidays
   - **Derived Features**: Rolling statistics, lag features

3. **Model Selection**
   - **Ensemble Methods**: Random Forest, XGBoost
   - **Deep Learning**: LSTM with attention mechanism
   - **Statistical**: ARIMA (baseline comparison)

4. **Training Strategy**
   - Train-test split: 85% / 15%
   - Window size: 14 days (optimized through experimentation)
   - Early stopping with patience=5
   - Loss function: MSE
   - Optimizer: NAdam

**LSTM Architecture Details**:
- Input: 14-day sliding window
- LSTM units: 32 (optimized)
- Attention layer: Improves performance on multivariate inputs
- Dropout: 0.2 for regularization
- Output: Next week's sales prediction

### Recommender System Architecture

The hybrid recommender combines two approaches:

1. **Collaborative Filtering (Item-Based)**
   - Uses Jaccard similarity for binary transaction data
   - Matrix factorization with SVD for dimensionality reduction
   - Handles sparsity in user-item matrix

2. **Content-Based Filtering**
   - Product features: Category, location, price range, order frequency
   - Clustering: Gaussian Mixture Model (3 components)
   - Similarity computation based on product attributes

3. **Hybrid Weighting**
   - 60% weight to collaborative filtering (captures user behavior)
   - 40% weight to content-based (ensures diversity)
   - Final ranking based on combined scores

4. **Clustering Strategy**
   - Cabinet grouping based on:
     - Location and segment
     - Product mix similarity
     - Sales performance
     - Cabinet age
   - Feature scaling: MinMaxScaler
   - Encoding: One-hot + Label encoding

---

##  Experimental Insights

### Forecasting Experiments

**Tested Variations**:
- Window sizes: 7, 14, 21, 28 days → **14 days optimal**
- LSTM units: 16, 32, 64 → **32 optimal** (64 increased errors)
- Optimizers: Adam, NAdam, RMSprop → **NAdam best**
- Loss functions: MSE, MAE → **MSE recommended**
- Batch sizes: 32, 64, 128 → **64 optimal**
- Epochs: 20, 30, 50 with early stopping → **30 with patience=5**

**Observations**:
- Bidirectional LSTM: Lower performance with 2x training time
- Adding hidden layers: Marginal improvement, significant time increase
- Attention mechanism: 8-12% improvement for multivariate
- Larger train size: Consistently improves results

### Recommender Experiments

**Distance Metrics**:
- Jaccard similarity vs. Cosine similarity → **Jaccard better for binary data**
- Matrix factorization: Tested but didn't improve sparse matrix handling

**Clustering**:
- K-Means vs. Gaussian Mixture → **GMM more robust**
- Cluster count: 2, 3, 4, 5 → **3 optimal** (balances granularity and overlap)

---

## 🔮 Future Enhancements

### Short-term Improvements
- [ ] **A/B Testing Framework**: Implement controlled experiments for model validation
- [ ] **Real-time Predictions**: Deploy API for live forecasting
- [ ] **Enhanced Visualizations**: Interactive dashboards with Plotly/Streamlit
- [ ] **Automated Retraining**: Schedule periodic model updates
- [ ] **Extended Feature Engineering**: Social events, promotions, competitor data

### Long-term Vision
- [ ] **Reinforcement Learning**: Dynamic inventory optimization
- [ ] **Multi-location Forecasting**: Cross-cabinet learning
- [ ] **Pricing Optimization**: Dynamic pricing based on demand
- [ ] **Customer Segmentation**: Personalized recommendations per user profile
- [ ] **Supply Chain Integration**: End-to-end optimization

### Model Improvements
- [ ] **Transformer Architecture**: Test attention-based models for longer sequences
- [ ] **Ensemble Stacking**: Combine LSTM + XGBoost predictions
- [ ] **Uncertainty Quantification**: Prediction intervals for risk management
- [ ] **Transfer Learning**: Apply pre-trained models across locations

---

## 📚 Data Schema

**Required Input Format**:

```python
# Transaction Data
columns = [
    'Timestamp',          # Transaction datetime
    'Date',              # Date only
    'DeviceId',          # Unique cabinet identifier
    'DeviceName',        # Cabinet name/location
    'barcode',           # Product barcode
    'name',              # Product name
    'NameClean',         # Cleaned product name
    'Category',          # Product category
    'NewLocation',       # Geographic location
    'locationName',      # Location name
    'Segment_1',         # Market segment
    'priceEuro',         # Price in EUR
    'TotalCount',        # Quantity sold
    'totalSumEuro',      # Total revenue
    'CustomerId',        # Customer ID (for recommender)
    'Status',            # Transaction status
    'Organization_Id'    # Organization identifier
]
```

**External Data** (optional):
- Weather: `Date`, `NewLocation`, `temperature`, `humidity`
- Holidays: `Country`, `date`, `holiday_name`

---

## 🧪 Testing

Run unit tests:
```bash
python -m pytest models_tests/
```

Validate preprocessing:
```bash
python -m tests_utils --validate-preprocessing
```

---

## 📈 MLflow Tracking

This project uses MLflow for experiment tracking:

```bash
# Start MLflow UI
mlflow ui

# View experiments at http://localhost:5000
```

**Logged Metrics**:
- Model parameters (hyperparameters)
- Training metrics (MSE, RMSE, MAE, MAPE)
- External parameters (window size, train size)
- Model artifacts

---

## 🤝 Contributing

This is a portfolio project, but suggestions and improvements are welcome!

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -am 'Add new feature'`)
4. Push to branch (`git push origin feature/improvement`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👤 About

**Created by**: Carlos Esteban De La Cruz Ribadeneira  
**Portfolio**: [GitHub Profile](https://github.com/tabers77)  
**LinkedIn**: [LinkedIn Profile](https://www.linkedin.com/in/carlosdlc/)  
**Email**: Contact via LinkedIn

### Project Context

This project was developed as part of a retail analytics initiative to demonstrate:
- End-to-end ML pipeline development
- Time series forecasting techniques
- Recommender system implementation
- Production-ready code structure
- Experiment tracking and model evaluation

**Tech Stack Expertise**: Python, TensorFlow, Scikit-learn, Pandas, MLflow, XGBoost

---

## 🙏 Acknowledgments

- Time series forecasting techniques inspired by state-of-the-art research
- Recommender system approaches based on collaborative filtering literature
- MLflow integration for professional experiment tracking

---

## 📞 Contact & Questions

For questions, suggestions, or collaboration opportunities:
- � LinkedIn: [Carlos De La Cruz](https://www.linkedin.com/in/carlosdlc/)
- 🐙 GitHub: [@tabers77](https://github.com/tabers77)

---

**⭐ If you find this project useful, please consider giving it a star!**
