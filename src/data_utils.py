"""
Data utilities for credit score classification.

This module provides functions for loading, cleaning, and preprocessing
credit score data extracted from exploratory notebooks.
"""

from typing import Tuple
import pandas as pd
import numpy as np


def load_data(path: str) -> pd.DataFrame:
    """
    Load credit score dataset from CSV file.
    
    Args:
        path: Path to CSV file containing credit score data
        
    Returns:
        DataFrame with raw credit score data
        
    Example:
        >>> df = load_data('../data/raw/set_credit_score.csv')
        >>> print(df.shape)
        (28000, 29)
    """
    df = pd.read_csv(path)
    print(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def basic_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform basic data cleaning operations.
    
    Handles:
    - Removes identifier columns (ID, Customer_ID, Name, SSN)
    - Strips trailing underscores from categorical values
    - Fills missing categorical values with 'Unknown'
    - Fills missing numeric values with median
    
    Args:
        df: Raw DataFrame with potential data quality issues
        
    Returns:
        Cleaned DataFrame ready for further preprocessing
        
    Example:
        >>> df_clean = basic_cleaning(df)
        >>> print(df_clean.isnull().sum().sum())
        0
    """
    df = df.copy()
    
    # Remove identifier columns that don't contribute to predictions
    id_columns = ['ID', 'Customer_ID', 'Name', 'SSN']
    existing_id_cols = [col for col in id_columns if col in df.columns]
    if existing_id_cols:
        df = df.drop(columns=existing_id_cols)
        print(f"Removed identifier columns: {existing_id_cols}")
    
    # Clean categorical columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        # Strip trailing underscores (common data quality issue)
        df[col] = df[col].str.strip('_')
        
        # Fill missing values
        if df[col].isnull().any():
            df[col] = df[col].fillna('Unknown')
            print(f"Filled {col} missing values with 'Unknown'")
    
    # Clean numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isnull().any():
            median_value = df[col].median()
            df[col] = df[col].fillna(median_value)
            print(f"Filled {col} missing values with median: {median_value:.2f}")
    
    print(f"Cleaning complete. Final shape: {df.shape}")
    return df


def remove_outliers_iqr(
    df: pd.DataFrame, 
    columns: list = None, 
    multiplier: float = 1.5
) -> pd.DataFrame:
    """
    Remove outliers using the Interquartile Range (IQR) method.
    
    For each numeric column, removes values outside the range:
    [Q1 - multiplier*IQR, Q3 + multiplier*IQR]
    
    Args:
        df: DataFrame to clean
        columns: List of column names to check for outliers.
                 If None, uses all numeric columns.
        multiplier: IQR multiplier (1.5 = standard, 3.0 = extreme outliers only)
        
    Returns:
        DataFrame with outliers removed
        
    Example:
        >>> df_no_outliers = remove_outliers_iqr(df, columns=['Annual_Income', 'Outstanding_Debt'])
        >>> print(f"Removed {len(df) - len(df_no_outliers)} outlier rows")
    """
    df = df.copy()
    initial_rows = len(df)
    
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    for col in columns:
        if col not in df.columns:
            continue
            
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR
        
        # Filter out outliers
        outlier_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
        outliers_removed = outlier_mask.sum()
        
        if outliers_removed > 0:
            df = df[~outlier_mask]
            print(f"Removed {outliers_removed} outliers from {col}")
    
    total_removed = initial_rows - len(df)
    print(f"Total rows removed: {total_removed} ({100*total_removed/initial_rows:.2f}%)")
    
    return df


def split_features_target(
    df: pd.DataFrame, 
    target_col: str = 'Credit_Score'
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Split DataFrame into features (X) and target (y).
    
    Args:
        df: Complete DataFrame including target column
        target_col: Name of the target variable column
        
    Returns:
        Tuple of (X, y) where X is features and y is target
        
    Example:
        >>> X, y = split_features_target(df)
        >>> print(f"Features: {X.shape}, Target: {y.shape}")
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame")
    
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Target distribution:\n{y.value_counts()}")
    
    return X, y


def get_feature_types(df: pd.DataFrame) -> dict:
    """
    Identify numeric and categorical features in DataFrame.
    
    Args:
        df: DataFrame to analyze
        
    Returns:
        Dictionary with 'numeric' and 'categorical' keys containing column lists
        
    Example:
        >>> feature_types = get_feature_types(X_train)
        >>> print(f"Numeric features: {len(feature_types['numeric'])}")
    """
    numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = df.select_dtypes(include=['object']).columns.tolist()
    
    return {
        'numeric': numeric_features,
        'categorical': categorical_features
    }
