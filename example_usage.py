"""
Quick usage example for credit score classification modules.

This script demonstrates how to use the reusable functions from src/
to load data, clean it, and train models.
"""

# Add src to path (if running from project root)
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from src.data_utils import load_data, basic_cleaning, split_features_target
from src.modeling import train_models, compare_models

def main():
    """Run the complete credit score classification pipeline."""
    
    print("="*60)
    print("CREDIT SCORE CLASSIFICATION - QUICK DEMO")
    print("="*60)
    
    # Step 1: Load data
    print("\n[STEP 1] Loading data...")
    df = load_data('data/raw/set_credit_score.csv')
    
    # Step 2: Basic cleaning
    print("\n[STEP 2] Cleaning data...")
    df_clean = basic_cleaning(df)
    
    # Step 3: Split features and target
    print("\n[STEP 3] Splitting features and target...")
    X, y = split_features_target(df_clean, target_col='Credit_Score')
    
    # Step 4: Train models (includes train/test split internally)
    print("\n[STEP 4] Training models with hyperparameter tuning...")
    print("(This may take several minutes...)")
    
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    results = train_models(X_train, y_train, X_test, y_test)
    
    # Step 5: Show comparison
    print("\n[STEP 5] Model Comparison:")
    comparison = compare_models(results['results'])
    print(comparison.to_string(index=False))
    
    print("\n" + "="*60)
    print(f"BEST MODEL: {results['best_model_name']}")
    print(f"Test Accuracy: {results['results'][results['best_model_name']]['test_accuracy']:.4f}")
    print("="*60)

if __name__ == "__main__":
    main()
