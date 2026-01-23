"""
ML Utilities for Surrogate Model Training
=========================================

Private helper module for Day 4's ML pipeline.
Handles data loading, cleaning, and feature engineering.
"""

import pandas as pd
import numpy as np
from pathlib import Path


def load_clean_df(path: str = "artifacts/results.csv") -> pd.DataFrame:
    """
    Load results CSV and clean for ML training.
    
    Steps:
    1. Load CSV
    2. Filter to only successful designs (ok == True)
    3. Drop rows with NaN values
    4. Ensure numeric dtypes
    
    Parameters
    ----------
    path : str
        Path to results CSV from Day 3
    
    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame ready for feature extraction
    """
    df = pd.read_csv(path)
    
    # Filter to successful designs only
    df = df[df['ok'] == True].copy()
    
    # Drop rows with any NaN in key columns
    key_cols = ['span', 'height', 'brace', 'sec_col', 'sec_beam', 'sec_brace',
                'udl_w', 'wind_P', 'drift', 'volume', 'carbon', 'max_abs_M']
    df = df.dropna(subset=key_cols)
    
    # Ensure numeric dtypes
    numeric_cols = ['span', 'height', 'brace', 'sec_col', 'sec_beam', 
                    'sec_brace', 'udl_w', 'wind_P', 'shipping_limit',
                    'drift', 'max_abs_M', 'volume', 'carbon']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Drop any rows that became NaN after type conversion
    df = df.dropna(subset=numeric_cols)
    
    print(f"Loaded {len(df)} clean designs from {path}")
    return df.reset_index(drop=True)


def make_features(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """
    Extract feature matrix X from DataFrame.
    
    Features:
    - span: bay width (m)
    - height: column height (m)
    - brace: bracing config (0 or 1)
    - sec_col: column section index
    - sec_beam: beam section index
    - sec_brace: brace section index
    - udl_w: gravity load (N/m, negative)
    - wind_P: lateral load (N)
    
    Parameters
    ----------
    df : pd.DataFrame
        Cleaned DataFrame from load_clean_df()
    
    Returns
    -------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features)
    feature_names : list[str]
        Names of features in order
    """
    feature_cols = [
        'span', 'height', 'brace', 
        'sec_col', 'sec_beam', 'sec_brace',
        'udl_w', 'wind_P'
    ]
    
    X = df[feature_cols].values.astype(np.float64)
    
    return X, feature_cols


def make_target(df: pd.DataFrame, target: str = "drift") -> np.ndarray:
    """
    Extract target vector y from DataFrame.
    
    Parameters
    ----------
    df : pd.DataFrame
        Cleaned DataFrame from load_clean_df()
    target : str
        Target column name. Options: 'drift', 'volume', 'carbon', 'max_abs_M'
    
    Returns
    -------
    y : np.ndarray
        Target vector of shape (n_samples,)
    """
    if target not in df.columns:
        raise ValueError(f"Target '{target}' not in DataFrame columns: {df.columns.tolist()}")
    
    y = df[target].values.astype(np.float64)
    return y


def params_to_features(params_list: list[dict]) -> np.ndarray:
    """
    Convert a list of parameter dicts to feature matrix.
    
    This is used in guided search to score candidate designs
    with the surrogate model before running the real solver.
    
    Parameters
    ----------
    params_list : list[dict]
        List of parameter dictionaries with keys matching feature_cols
    
    Returns
    -------
    X : np.ndarray
        Feature matrix of shape (n_candidates, n_features)
    """
    feature_cols = [
        'span', 'height', 'brace', 
        'sec_col', 'sec_beam', 'sec_brace',
        'udl_w', 'wind_P'
    ]
    
    X = np.array([[p[col] for col in feature_cols] for p in params_list], dtype=np.float64)
    return X