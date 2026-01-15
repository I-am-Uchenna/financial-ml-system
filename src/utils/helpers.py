import numpy as np
import pandas as pd
from typing import Tuple

def calculate_returns(prices: pd.Series, periods: int = 1) -> pd.Series:
    return prices.pct_change(periods=periods)

def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.02, 
                          periods_per_year: int = 252) -> float:
    excess_returns = returns - risk_free_rate / periods_per_year
    return np.sqrt(periods_per_year) * excess_returns.mean() / excess_returns.std()

def calculate_max_drawdown(prices: pd.Series) -> float:
    cumulative = (1 + calculate_returns(prices)).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    return drawdown.min()

def time_series_split(data: pd.DataFrame, test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
    split_idx = int(len(data) * (1 - test_size))
    return data.iloc[:split_idx], data.iloc[split_idx:]
