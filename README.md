# Financial ML System

Institutional-grade market prediction system combining SVM and Neural Networks.

## Notebooks

Run in sequence:
1. `01_data_pipeline.ipynb` - Data ingestion and processing
2. `02_feature_engineering.ipynb` - Technical indicators
3. `03_svm_training.ipynb` - Market regime classification
4. `04_nn_training.ipynb` - Return prediction
5. `05_signal_generation.ipynb` - Trading signals
6. `06_backtesting.ipynb` - Performance analysis

## Quick Start in Colab

```python
!git clone https://github.com/I-am-Uchenna/financial-ml-system.git
%cd financial-ml-system
!pip install -r requirements.txt
```

Then open any notebook from the `notebooks/` directory.
