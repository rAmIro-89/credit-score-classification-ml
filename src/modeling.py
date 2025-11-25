"""
Model training and evaluation utilities for credit score classification.

This module provides functions for training multiple models with hyperparameter
tuning and generating comprehensive performance reports.
"""

from typing import Dict, Tuple, Any
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, 
    classification_report, 
    confusion_matrix
)


def prepare_train_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split data into training and test sets with optional stratification.
    
    Args:
        X: Feature DataFrame
        y: Target Series
        test_size: Proportion of data to use for testing (0.0 to 1.0)
        random_state: Random seed for reproducibility
        stratify: Whether to maintain class distribution in splits
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
        
    Example:
        >>> X_train, X_test, y_train, y_test = prepare_train_test_split(X, y)
        >>> print(f"Training set: {len(X_train)}, Test set: {len(X_test)}")
    """
    stratify_param = y if stratify else None
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state, 
        stratify=stratify_param
    )
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    print(f"Class distribution in train set:\n{y_train.value_counts()}")
    
    return X_train, X_test, y_train, y_test


def scale_features(
    X_train: pd.DataFrame, 
    X_test: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    """
    Standardize features using StandardScaler (fit on train, transform on test).
    
    Important: Only use scaled data for distance-based models (KNN, SVM).
    Tree-based models (Decision Tree, Random Forest) don't require scaling.
    
    Args:
        X_train: Training features
        X_test: Test features
        
    Returns:
        Tuple of (X_train_scaled, X_test_scaled, scaler)
        
    Example:
        >>> X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
        >>> print(f"Scaled train mean: {X_train_scaled.mean():.4f}")
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"Features scaled. Train mean: {X_train_scaled.mean():.4f}, std: {X_train_scaled.std():.4f}")
    
    return X_train_scaled, X_test_scaled, scaler


def train_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Train Decision Tree, Random Forest, and KNN with hyperparameter tuning.
    
    Uses GridSearchCV with 5-fold cross-validation to find optimal parameters
    for each model. KNN uses standardized features automatically.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        random_state: Random seed for reproducibility
        
    Returns:
        Dictionary containing:
        - 'models': Dict of trained model objects
        - 'results': Dict of performance metrics per model
        - 'best_model_name': Name of best-performing model
        
    Example:
        >>> results = train_models(X_train, y_train, X_test, y_test)
        >>> print(f"Best model: {results['best_model_name']}")
        >>> print(f"Test accuracy: {results['results'][results['best_model_name']]['test_accuracy']:.4f}")
    """
    print("="*60)
    print("TRAINING MODELS WITH HYPERPARAMETER TUNING")
    print("="*60)
    
    models = {}
    results = {}
    
    # Scale features for KNN (fit on train only)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # -------------------------
    # 1. DECISION TREE
    # -------------------------
    print("\n[1/3] Training Decision Tree...")
    param_grid_dt = {
        'max_depth': [10, 15, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    grid_dt = GridSearchCV(
        DecisionTreeClassifier(random_state=random_state),
        param_grid_dt,
        cv=5,
        scoring='accuracy',
        n_jobs=-1
    )
    grid_dt.fit(X_train, y_train)
    
    y_pred_dt = grid_dt.predict(X_test)
    test_acc_dt = accuracy_score(y_test, y_pred_dt)
    
    models['DecisionTree'] = grid_dt.best_estimator_
    results['DecisionTree'] = {
        'best_params': grid_dt.best_params_,
        'cv_accuracy': grid_dt.best_score_,
        'test_accuracy': test_acc_dt,
        'predictions': y_pred_dt
    }
    
    print(f"  ✓ CV Accuracy: {grid_dt.best_score_:.4f}")
    print(f"  ✓ Test Accuracy: {test_acc_dt:.4f}")
    print(f"  ✓ Best params: {grid_dt.best_params_}")
    
    # -------------------------
    # 2. RANDOM FOREST
    # -------------------------
    print("\n[2/3] Training Random Forest...")
    param_grid_rf = {
        'n_estimators': [100, 200],
        'max_depth': [15, 20, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
    
    grid_rf = GridSearchCV(
        RandomForestClassifier(random_state=random_state),
        param_grid_rf,
        cv=5,
        scoring='accuracy',
        n_jobs=-1
    )
    grid_rf.fit(X_train, y_train)
    
    y_pred_rf = grid_rf.predict(X_test)
    test_acc_rf = accuracy_score(y_test, y_pred_rf)
    
    models['RandomForest'] = grid_rf.best_estimator_
    results['RandomForest'] = {
        'best_params': grid_rf.best_params_,
        'cv_accuracy': grid_rf.best_score_,
        'test_accuracy': test_acc_rf,
        'predictions': y_pred_rf
    }
    
    print(f"  ✓ CV Accuracy: {grid_rf.best_score_:.4f}")
    print(f"  ✓ Test Accuracy: {test_acc_rf:.4f}")
    print(f"  ✓ Best params: {grid_rf.best_params_}")
    
    # -------------------------
    # 3. K-NEAREST NEIGHBORS
    # -------------------------
    print("\n[3/3] Training KNN (using scaled features)...")
    param_grid_knn = {
        'n_neighbors': [3, 5, 7, 9],
        'weights': ['uniform', 'distance'],
        'p': [1, 2]  # 1=Manhattan, 2=Euclidean
    }
    
    grid_knn = GridSearchCV(
        KNeighborsClassifier(),
        param_grid_knn,
        cv=5,
        scoring='accuracy',
        n_jobs=-1
    )
    grid_knn.fit(X_train_scaled, y_train)
    
    y_pred_knn = grid_knn.predict(X_test_scaled)
    test_acc_knn = accuracy_score(y_test, y_pred_knn)
    
    models['KNN'] = {
        'model': grid_knn.best_estimator_,
        'scaler': scaler  # Store scaler for KNN predictions
    }
    results['KNN'] = {
        'best_params': grid_knn.best_params_,
        'cv_accuracy': grid_knn.best_score_,
        'test_accuracy': test_acc_knn,
        'predictions': y_pred_knn
    }
    
    print(f"  ✓ CV Accuracy: {grid_knn.best_score_:.4f}")
    print(f"  ✓ Test Accuracy: {test_acc_knn:.4f}")
    print(f"  ✓ Best params: {grid_knn.best_params_}")
    
    # -------------------------
    # DETERMINE BEST MODEL
    # -------------------------
    best_model_name = max(results.keys(), key=lambda k: results[k]['test_accuracy'])
    
    print("\n" + "="*60)
    print(f"🏆 BEST MODEL: {best_model_name}")
    print(f"   Test Accuracy: {results[best_model_name]['test_accuracy']:.4f}")
    print("="*60)
    
    return {
        'models': models,
        'results': results,
        'best_model_name': best_model_name
    }


def evaluate_model(
    y_true: pd.Series, 
    y_pred: np.ndarray, 
    model_name: str = "Model"
) -> Dict[str, Any]:
    """
    Generate comprehensive evaluation metrics for a classification model.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        model_name: Name of model for display purposes
        
    Returns:
        Dictionary with accuracy, classification report, and confusion matrix
        
    Example:
        >>> metrics = evaluate_model(y_test, predictions, "Random Forest")
        >>> print(metrics['classification_report'])
    """
    print(f"\n{'='*60}")
    print(f"EVALUATION: {model_name}")
    print(f"{'='*60}")
    
    # Accuracy
    accuracy = accuracy_score(y_true, y_pred)
    print(f"\nOverall Accuracy: {accuracy:.4f}")
    
    # Classification Report
    report = classification_report(y_true, y_pred, output_dict=True)
    print(f"\nClassification Report:")
    print(classification_report(y_true, y_pred))
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    print(f"\nConfusion Matrix:")
    print(cm)
    
    return {
        'accuracy': accuracy,
        'classification_report': report,
        'confusion_matrix': cm
    }


def compare_models(results: Dict[str, Dict]) -> pd.DataFrame:
    """
    Create comparison table of model performances.
    
    Args:
        results: Dictionary from train_models() containing model results
        
    Returns:
        DataFrame with model comparison metrics
        
    Example:
        >>> comparison = compare_models(training_results['results'])
        >>> print(comparison)
    """
    comparison_data = []
    
    for model_name, metrics in results.items():
        comparison_data.append({
            'Model': model_name,
            'CV Accuracy': f"{metrics['cv_accuracy']:.4f}",
            'Test Accuracy': f"{metrics['test_accuracy']:.4f}",
            'Best Params': str(metrics['best_params'])
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df = comparison_df.sort_values('Test Accuracy', ascending=False)
    
    return comparison_df
