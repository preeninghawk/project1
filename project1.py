import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix,
                             precision_score, recall_score, f1_score,
                             precision_recall_curve, roc_auc_score, roc_curve)
from sklearn.impute import SimpleImputer
import warnings

warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)


# ==============================================================================
# PROBLEM 1: Calculate profits under different scenarios
# ==============================================================================

def calculate_profit(y_true, y_pred, P=100, Q=1000, S=2000):
    """
    Calculate total profit based on predictions

    Args:
        y_true: actual labels (0=normal, 1=defect)
        y_pred: predicted labels (0=normal, 1=defect)
        P: production cost
        Q: test cost
        S: sale price

    Returns:
        total_profit: total profit in won
    """
    TP = np.sum((y_pred == 1) & (y_true == 1))  # Correctly identified defects
    TN = np.sum((y_pred == 0) & (y_true == 0))  # Correctly identified normals
    FP = np.sum((y_pred == 1) & (y_true == 0))  # Normal wrongly marked as defect
    FN = np.sum((y_pred == 0) & (y_true == 1))  # Defect wrongly marked as normal

    # Profit calculation:
    # TP: Defects discarded without test -> lose P (100)
    # TN: Normals tested and sold -> profit (S - P - Q) = 900
    # FP: Normals discarded -> lose (S - P - Q + P) = lose potential profit + P
    # FN: Defects tested -> lose (P + Q) = 1100

    profit_TP = TP * (-P)  # Discarded defects: -100 each
    profit_TN = TN * (S - P - Q)  # Sold normals: +900 each
    profit_FP = FP * (-P)  # Wrongly discarded normals: -100 each (but lost 900 profit)
    profit_FN = FN * (-(P + Q))  # Tested defects: -1100 each

    total_profit = profit_TP + profit_TN + profit_FP + profit_FN

    return total_profit


def problem1_analysis(y_train, P=100, Q=1000, S=2000):
    """
    Problem 1: Calculate profits for different scenarios
    """
    print("=" * 80)
    print("PROBLEM 1: Profit Analysis")
    print("=" * 80)

    n_total = len(y_train)
    n_normal = np.sum(y_train == 0)
    n_defect = np.sum(y_train == 1)
    defect_rate = n_defect / n_total

    print(f"\nDataset Statistics:")
    print(f"Total samples: {n_total}")
    print(f"Normal: {n_normal} ({n_normal / n_total * 100:.2f}%)")
    print(f"Defect: {n_defect} ({n_defect / n_total * 100:.2f}%)")

    # Scenario 1: Test all chips (no model)
    profit_no_model = n_normal * (S - P - Q) + n_defect * (-(P + Q))
    print(f"\n1) No model (test all): {profit_no_model:,} won")

    # Scenario 2: Perfect model
    profit_perfect = n_normal * (S - P - Q) + n_defect * (-P)
    print(f"2) Perfect model: {profit_perfect:,} won")
    print(f"   Improvement: {profit_perfect - profit_no_model:,} won")

    # Scenario 3: 90% Precision and 90% Recall model
    recall_90 = 0.90
    precision_90 = 0.90

    # With 90% recall: we correctly identify 90% of defects
    TP_90 = int(n_defect * recall_90)
    FN_90 = n_defect - TP_90

    # With 90% precision: among predicted defects, 90% are actually defects
    # Precision = TP / (TP + FP) = 0.9
    # So: TP = 0.9 * (TP + FP)
    # FP = TP / 0.9 - TP = TP * (1/0.9 - 1)
    FP_90 = int(TP_90 * (1 / precision_90 - 1))
    TN_90 = n_normal - FP_90

    profit_90 = (TP_90 * (-P) + TN_90 * (S - P - Q) +
                 FP_90 * (-P) + FN_90 * (-(P + Q)))

    print(f"3) 90% Recall & 90% Precision model: {profit_90:,} won")
    print(f"   TP: {TP_90}, TN: {TN_90}, FP: {FP_90}, FN: {FN_90}")
    print(f"   Improvement over no model: {profit_90 - profit_no_model:,} won")

    return profit_no_model, profit_perfect, profit_90


# ==============================================================================
# PROBLEM 2: EDA and Preprocessing
# ==============================================================================

def load_and_explore_data(train_path):
    """
    Load data and perform initial exploration
    """
    print("\n" + "=" * 80)
    print("PROBLEM 2: EDA and Preprocessing")
    print("=" * 80)

    # Load data
    df_train = pd.read_csv(train_path)

    print(f"\nOriginal data shape: {df_train.shape}")
    print(f"Columns: Label + {df_train.shape[1] - 1} sensor variables")

    # Separate features and labels
    y = df_train['Label'].map({'normal': 0, 'defect': 1})
    X = df_train.drop('Label', axis=1)

    return X, y, df_train


def comprehensive_eda(X, y):
    """
    Perform comprehensive EDA
    """
    print("\n--- EDA: Missing Values ---")
    missing_counts = X.isnull().sum()
    missing_pct = (missing_counts / len(X)) * 100
    missing_df = pd.DataFrame({
        'Missing_Count': missing_counts,
        'Missing_Percentage': missing_pct
    }).sort_values('Missing_Count', ascending=False)

    print(f"Variables with missing values: {(missing_counts > 0).sum()}")
    print("\nTop 10 variables with most missing values:")
    print(missing_df.head(10))

    # Check for zero-variance features
    print("\n--- EDA: Zero-Variance Features ---")
    zero_var_cols = []
    for col in X.columns:
        if X[col].dtype in ['int64', 'float64']:
            if X[col].std() == 0 or X[col].nunique() == 1:
                zero_var_cols.append(col)

    print(f"Zero-variance features: {len(zero_var_cols)}")
    if len(zero_var_cols) > 0:
        print(f"Examples: {zero_var_cols[:5]}")

    # Check data types
    print("\n--- EDA: Data Types ---")
    print(X.dtypes.value_counts())

    # Class distribution
    print("\n--- EDA: Class Distribution ---")
    print(y.value_counts())
    print(f"Imbalance ratio: {y.value_counts()[0] / y.value_counts()[1]:.2f}:1")

    return missing_df, zero_var_cols


def preprocess_data(X, y, missing_df=None, zero_var_cols=None,
                    train_imputer=None, train_scaler=None,
                    zero_var_fill_map=None, is_train=True):
    """
    Preprocess data with proper handling for train/test

    Args:
        X: features
        y: labels (None for test data)
        missing_df: missing value statistics from training
        zero_var_cols: zero variance columns from training
        train_imputer: fitted imputer from training
        train_scaler: fitted scaler from training
        zero_var_fill_map: mapping of zero-var columns to fill values (0 or 1)
        is_train: whether this is training data
    """
    X_processed = X.copy()

    # Step 1: Handle zero-variance columns - fill NAs based on dominant value
    if is_train and zero_var_cols is not None and len(zero_var_cols) > 0:
        zero_var_fill_map = {}
        print(f"\n--- Handling Zero-Variance Columns ---")
        print(f"Found {len(zero_var_cols)} zero-variance columns")

        for col in zero_var_cols:
            if col in X_processed.columns:
                # Get non-null values
                non_null_values = X_processed[col].dropna()

                if len(non_null_values) > 0:
                    # Count zeros and non-zeros
                    zero_count = (non_null_values == 0).sum()
                    non_zero_count = (non_null_values != 0).sum()

                    # If dominantly zero, fill NAs with 1, otherwise fill with 0
                    if zero_count > non_zero_count:
                        fill_value = 1
                    else:
                        fill_value = 0

                    zero_var_fill_map[col] = fill_value
                    X_processed[col].fillna(fill_value, inplace=True)
                else:
                    # If all values are null, fill with 0
                    zero_var_fill_map[col] = 0
                    X_processed[col].fillna(0, inplace=True)

    elif not is_train and zero_var_fill_map is not None:
        # For test data: apply same fill strategy, then remove columns
        for col, fill_value in zero_var_fill_map.items():
            if col in X_processed.columns:
                X_processed[col].fillna(fill_value, inplace=True)

    # Step 2: Handle missing values for remaining columns
    if is_train:
        # For training: fit imputer based on missing percentage
        high_missing_cols = missing_df[missing_df['Missing_Percentage'] > 50].index.tolist()
        # Only consider columns that still exist after zero-var removal
        high_missing_cols = [col for col in high_missing_cols if col in X_processed.columns]

        low_missing_cols = [col for col in X_processed.columns
                            if col not in high_missing_cols and X_processed[col].isnull().any()]

        print(f"\n--- Missing Value Handling (Non-Zero-Variance Columns) ---")
        print(f"  - High missing (>50%, fill with 0): {len(high_missing_cols)} columns")
        print(f"  - Low missing (≤50%, fill with mean): {len(low_missing_cols)} columns")

        # Fill high missing with 0
        for col in high_missing_cols:
            X_processed[col].fillna(0, inplace=True)

        # Fill low missing with mean
        imputer = SimpleImputer(strategy='mean')
        if low_missing_cols:
            X_processed[low_missing_cols] = imputer.fit_transform(X_processed[low_missing_cols])

        train_imputer = {'high_missing': high_missing_cols,
                         'low_missing': low_missing_cols,
                         'imputer': imputer}
    else:
        # For test: use training imputer
        high_missing_cols = train_imputer['high_missing']
        low_missing_cols = train_imputer['low_missing']
        imputer = train_imputer['imputer']

        for col in high_missing_cols:
            if col in X_processed.columns:
                X_processed[col].fillna(0, inplace=True)

        if low_missing_cols:
            cols_to_impute = [col for col in low_missing_cols if col in X_processed.columns]
            if cols_to_impute:
                X_processed[cols_to_impute] = imputer.transform(X_processed[cols_to_impute])

    # Step 3: Standardization
    if is_train:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_processed)
        train_scaler = scaler
    else:
        X_scaled = train_scaler.transform(X_processed)

    X_final = pd.DataFrame(X_scaled, columns=X_processed.columns, index=X_processed.index)

    if is_train:
        print(f"\nFinal preprocessed data shape: {X_final.shape}")
        print(f"Samples: {X_final.shape[0]}, Features: {X_final.shape[1]}")
        return X_final, y, train_imputer, train_scaler, zero_var_fill_map
    else:
        return X_final


# ==============================================================================
# PROBLEM 3: Model Development and Selection
# ==============================================================================

def find_optimal_threshold(y_true, y_proba, target_precision=0.90, P=100, Q=1000, S=2000):
    """
    Find optimal threshold balancing precision target and profit
    """
    thresholds = np.arange(0.01, 0.99, 0.01)
    results = []

    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)

        if np.sum(y_pred) == 0:  # No defects predicted
            continue

        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        profit = calculate_profit(y_true, y_pred, P, Q, S)

        results.append({
            'threshold': threshold,
            'precision': precision,
            'recall': recall,
            'profit': profit
        })

    results_df = pd.DataFrame(results)

    # Find threshold with precision >= target_precision and maximum profit
    high_precision = results_df[results_df['precision'] >= target_precision]

    if len(high_precision) > 0:
        best_idx = high_precision['profit'].idxmax()
        best_threshold = results_df.loc[best_idx, 'threshold']
        best_precision = results_df.loc[best_idx, 'precision']
        best_recall = results_df.loc[best_idx, 'recall']
        best_profit = results_df.loc[best_idx, 'profit']
    else:
        # If no threshold achieves target precision, pick highest precision
        best_idx = results_df['precision'].idxmax()
        best_threshold = results_df.loc[best_idx, 'threshold']
        best_precision = results_df.loc[best_idx, 'precision']
        best_recall = results_df.loc[best_idx, 'recall']
        best_profit = results_df.loc[best_idx, 'profit']
        print(f"Warning: Could not achieve {target_precision:.0%} precision. Best: {best_precision:.4f}")

    return best_threshold, best_precision, best_recall, best_profit, results_df


def train_and_evaluate_models(X_train, y_train, target_precision=0.90):
    """
    Train multiple models and select the best one
    """
    print("\n" + "=" * 80)
    print("PROBLEM 3: Model Development")
    print("=" * 80)

    # Split for validation
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )

    print(f"\nTrain set: {X_tr.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples")

    models = {
        'Logistic Regression (L2)': LogisticRegression(
            penalty='l2', C=1.0, max_iter=1000, random_state=42, class_weight='balanced'
        ),
        'Logistic Regression (L2, C=0.1)': LogisticRegression(
            penalty='l2', C=0.1, max_iter=1000, random_state=42, class_weight='balanced'
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=100, max_depth=10, random_state=42, class_weight='balanced'
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=100, max_depth=5, random_state=42
        )
    }

    results = {}

    for name, model in models.items():
        print(f"\n--- Training: {name} ---")

        # Train
        model.fit(X_tr, y_tr)

        # Predict probabilities
        y_tr_proba = model.predict_proba(X_tr)[:, 1]
        y_val_proba = model.predict_proba(X_val)[:, 1]

        # Find optimal threshold
        threshold, precision, recall, profit, _ = find_optimal_threshold(
            y_val, y_val_proba, target_precision
        )

        # Validation predictions with optimal threshold
        y_val_pred = (y_val_proba >= threshold).astype(int)

        # Calculate metrics
        val_precision = precision_score(y_val, y_val_pred)
        val_recall = recall_score(y_val, y_val_pred)
        val_f1 = f1_score(y_val, y_val_pred)
        val_auc = roc_auc_score(y_val, y_val_proba)
        val_profit = calculate_profit(y_val, y_val_pred)

        print(f"Optimal threshold: {threshold:.3f}")
        print(f"Validation - Precision: {val_precision:.4f}, Recall: {val_recall:.4f}")
        print(f"Validation - F1: {val_f1:.4f}, AUC: {val_auc:.4f}")
        print(f"Validation - Profit: {val_profit:,} won")

        results[name] = {
            'model': model,
            'threshold': threshold,
            'val_precision': val_precision,
            'val_recall': val_recall,
            'val_f1': val_f1,
            'val_auc': val_auc,
            'val_profit': val_profit
        }

    # Select best model based on validation profit
    best_model_name = max(results, key=lambda x: results[x]['val_profit'])
    best_model_info = results[best_model_name]

    print(f"\n{'=' * 80}")
    print(f"BEST MODEL: {best_model_name}")
    print(f"{'=' * 80}")
    print(f"Threshold: {best_model_info['threshold']:.3f}")
    print(f"Validation Precision: {best_model_info['val_precision']:.4f}")
    print(f"Validation Recall: {best_model_info['val_recall']:.4f}")
    print(f"Validation F1: {best_model_info['val_f1']:.4f}")
    print(f"Validation Profit: {best_model_info['val_profit']:,} won")

    return best_model_info['model'], best_model_info['threshold'], results, X_val, y_val


# ==============================================================================
# PROBLEM 4: Predict on test data
# ==============================================================================

def predict_test_data(model, threshold, X_test, output_file):
    """
    Predict on test data and save results
    """
    print("\n" + "=" * 80)
    print("PROBLEM 4: Test Data Prediction")
    print("=" * 80)

    y_test_proba = model.predict_proba(X_test)[:, 1]
    y_test_pred = (y_test_proba >= threshold).astype(int)

    # Save predictions
    with open(output_file, 'w') as f:
        for pred in y_test_pred:
            f.write(f"{pred}\n")

    print(f"\nPredictions saved to: {output_file}")
    print(f"Total predictions: {len(y_test_pred)}")
    print(f"Predicted normal: {np.sum(y_test_pred == 0)}")
    print(f"Predicted defect: {np.sum(y_test_pred == 1)}")

    return y_test_pred


# ==============================================================================
# PROBLEM 5: Evaluate model on full training data
# ==============================================================================

def evaluate_on_full_training(model, threshold, X_train, y_train,
                              profit_no_model, profit_perfect):
    """
    Evaluate model on full training data
    """
    print("\n" + "=" * 80)
    print("PROBLEM 5: Model Evaluation on Training Data")
    print("=" * 80)

    y_train_proba = model.predict_proba(X_train)[:, 1]
    y_train_pred = (y_train_proba >= threshold).astype(int)

    precision = precision_score(y_train, y_train_pred)
    recall = recall_score(y_train, y_train_pred)
    f1 = f1_score(y_train, y_train_pred)

    profit_model = calculate_profit(y_train, y_train_pred)
    improvement = profit_model - profit_no_model
    pct_of_perfect = (profit_model - profit_no_model) / (profit_perfect - profit_no_model) * 100

    print(f"\nModel Performance:")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    print(f"\nProfit Comparison:")
    print(f"No model profit: {profit_no_model:,} won")
    print(f"Model profit: {profit_model:,} won")
    print(f"Perfect model profit: {profit_perfect:,} won")
    print(f"\nImprovement over no model: {improvement:,} won")
    print(f"Percentage of perfect model gain: {pct_of_perfect:.2f}%")

    # Confusion matrix
    cm = confusion_matrix(y_train, y_train_pred)
    print(f"\nConfusion Matrix:")
    print(f"                Predicted Normal  Predicted Defect")
    print(f"Actual Normal          {cm[0, 0]:6d}          {cm[0, 1]:6d}")
    print(f"Actual Defect          {cm[1, 0]:6d}          {cm[1, 1]:6d}")

    return profit_model


# ==============================================================================
# PROBLEM 6 & 7: Price change scenarios
# ==============================================================================

def optimize_for_new_price(model, X_train, y_train, S_new, P=100, Q=1000):
    """
    Optimize model for new selling price
    """
    print(f"\n--- New Price Scenario: S = {S_new} won ---")

    # Calculate baseline profits
    y_all_test = np.zeros(len(y_train))  # Predict all as normal (test all)
    profit_no_model = calculate_profit(y_train, y_all_test, P, Q, S_new)

    y_perfect = y_train.copy()
    profit_perfect = calculate_profit(y_train, y_perfect, P, Q, S_new)

    print(f"\nNo model profit: {profit_no_model:,} won")
    print(f"Perfect model profit: {profit_perfect:,} won")

    # Find optimal threshold for new price
    y_proba = model.predict_proba(X_train)[:, 1]

    thresholds = np.arange(0.01, 0.99, 0.01)
    best_profit = float('-inf')
    best_threshold = 0.5
    best_precision = 0
    best_recall = 0

    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        profit = calculate_profit(y_train, y_pred, P, Q, S_new)

        if profit > best_profit:
            best_profit = profit
            best_threshold = threshold
            if np.sum(y_pred) > 0:
                best_precision = precision_score(y_train, y_pred)
                best_recall = recall_score(y_train, y_pred)

    print(f"\nOptimized Model:")
    print(f"Best threshold: {best_threshold:.3f}")
    print(f"Precision: {best_precision:.4f}")
    print(f"Recall: {best_recall:.4f}")
    print(f"Model profit: {best_profit:,} won")
    print(f"Improvement over no model: {best_profit - profit_no_model:,} won")
    print(
        f"Percentage of perfect gain: {(best_profit - profit_no_model) / (profit_perfect - profit_no_model) * 100:.2f}%")

    return best_threshold, best_profit


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

def main():
    """
    Main execution function
    """
    # File paths - UPDATE THESE
    train_path = 'data98_semi_train.csv'
    test_path = 'data98_semi_test.csv'
    output_file = 'project01_YOUR_STUDENT_ID.txt'  # UPDATE with your student ID

    # Load and explore data
    X, y, df_train = load_and_explore_data(train_path)

    # EDA
    missing_df, zero_var_cols = comprehensive_eda(X, y)

    # Problem 1
    profit_no_model, profit_perfect, profit_90 = problem1_analysis(y)

    # Preprocess training data
    X_processed, y_processed, train_imputer, train_scaler, zero_var_fill_map = preprocess_data(
        X, y, missing_df, zero_var_cols, is_train=True
    )

    # Problem 3: Train and select model
    best_model, best_threshold, all_results, X_val, y_val = train_and_evaluate_models(
        X_processed, y_processed, target_precision=0.90
    )

    # Retrain on full training data
    print("\n--- Retraining on full training data ---")
    best_model.fit(X_processed, y_processed)

    # Problem 5: Evaluate on training data
    profit_model = evaluate_on_full_training(
        best_model, best_threshold, X_processed, y_processed,
        profit_no_model, profit_perfect
    )

    # Load and preprocess test data
    print("\n--- Loading test data ---")
    df_test = pd.read_csv(test_path)
    X_test = df_test.copy()

    # Preprocess test data using training statistics
    X_test_processed = preprocess_data(
        X_test, None, missing_df, zero_var_cols,
        train_imputer, train_scaler, zero_var_fill_map, is_train=False
    )

    # Problem 4: Predict on test data
    y_test_pred = predict_test_data(best_model, best_threshold, X_test_processed, output_file)

    # Problem 6: Price drops to 1,200 won
    print("\n" + "=" * 80)
    print("PROBLEM 6: Price Change to S = 1,200 won")
    print("=" * 80)
    threshold_1200, profit_1200 = optimize_for_new_price(
        best_model, X_processed, y_processed, S_new=1200
    )

    # Problem 7: Price increases to 5,000 won
    print("\n" + "=" * 80)
    print("PROBLEM 7: Price Change to S = 5,000 won")
    print("=" * 80)
    threshold_5000, profit_5000 = optimize_for_new_price(
        best_model, X_processed, y_processed, S_new=5000
    )

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print("\nSummary:")
    print(f"1. Original model (S=2000): Threshold={best_threshold:.3f}, Profit={profit_model:,}")
    print(f"2. Optimized for S=1200: Threshold={threshold_1200:.3f}, Profit={profit_1200:,}")
    print(f"3. Optimized for S=5000: Threshold={threshold_5000:.3f}, Profit={profit_5000:,}")


if __name__ == "__main__":
    main()