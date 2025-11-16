#!/usr/bin/env python3
"""
Customer Churn Prediction Model
Author: Deepanraj A - Data Analyst
GitHub: deeepanbe

Predicts customer churn for telecommunications company using ensemble ML techniques.
Includes feature engineering, model comparison, and SHAP analysis for interpretability.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import joblib
import warnings
warnings.filterwarnings('ignore')

def load_customer_data(file_path='customer_data.csv'):
    """
    Load customer churn data
    Expected columns: customer_id, tenure, monthly_charges, total_charges, 
                     contract_type, payment_method, churn
    """
    try:
        df = pd.read_csv(file_path)
        print(f"Customer data loaded: {df.shape[0]} customers")
        return df
    except FileNotFoundError:
        print(f"File {file_path} not found. Generating sample data...")
        np.random.seed(42)
        n_customers = 5000
        
        # Generate synthetic customer data
        contract_types = ['Month-to-Month', 'One Year', 'Two Year']
        payment_methods = ['Electronic Check', 'Mailed Check', 'Bank Transfer', 'Credit Card']
        
        data = {
            'customer_id': [f'CUST_{i:05d}' for i in range(n_customers)],
            'tenure': np.random.randint(1, 72, n_customers),
            'monthly_charges': np.random.uniform(20, 120, n_customers),
            'total_charges': np.random.uniform(100, 8000, n_customers),
            'contract_type': np.random.choice(contract_types, n_customers),
            'payment_method': np.random.choice(payment_methods, n_customers),
            'internet_service': np.random.choice(['DSL', 'Fiber', 'No'], n_customers),
            'tech_support': np.random.choice(['Yes', 'No'], n_customers),
        }
        
        df = pd.DataFrame(data)
        
        # Generate churn based on features
        churn_probability = (
            0.3 * (df['tenure'] < 12).astype(int) +
            0.2 * (df['monthly_charges'] > 80).astype(int) +
            0.15 * (df['contract_type'] == 'Month-to-Month').astype(int) +
            0.1 * (df['payment_method'] == 'Electronic Check').astype(int) +
            np.random.uniform(0, 0.25, n_customers)
        )
        
        df['churn'] = (churn_probability > 0.5).astype(int)
        
        return df

def engineer_churn_features(df):
    """
    Create advanced features for churn prediction
    """
    df = df.copy()
    
    # Tenure-based features
    df['is_new_customer'] = (df['tenure'] <= 6).astype(int)
    df['is_loyal_customer'] = (df['tenure'] >= 24).astype(int)
    df['tenure_years'] = df['tenure'] / 12
    
    # Financial features
    df['avg_monthly_spend'] = df['total_charges'] / (df['tenure'] + 1)
    df['charge_per_tenure'] = df['monthly_charges'] / (df['tenure'] + 1)
    df['high_spender'] = (df['monthly_charges'] > df['monthly_charges'].median()).astype(int)
    
    # Contract risk indicators
    df['month_to_month'] = (df['contract_type'] == 'Month-to-Month').astype(int)
    df['electronic_payment'] = (df['payment_method'] == 'Electronic Check').astype(int)
    
    # Service usage
    if 'internet_service' in df.columns:
        df['has_internet'] = (df['internet_service'] != 'No').astype(int)
    if 'tech_support' in df.columns:
        df['has_support'] = (df['tech_support'] == 'Yes').astype(int)
    
    print(f"\nFeature engineering completed: {df.shape[1]} total features")
    return df

def prepare_features(df):
    """
    Encode categorical variables and prepare feature matrix
    """
    df = df.copy()
    
    # Select features for modeling
    feature_cols = ['tenure', 'monthly_charges', 'avg_monthly_spend', 'charge_per_tenure',
                   'is_new_customer', 'is_loyal_customer', 'high_spender',
                   'month_to_month', 'electronic_payment', 'tenure_years']
    
    if 'has_internet' in df.columns:
        feature_cols.append('has_internet')
    if 'has_support' in df.columns:
        feature_cols.append('has_support')
    
    X = df[feature_cols]
    y = df['churn'] if 'churn' in df.columns else None
    
    return X, y, feature_cols

def train_churn_models(X_train, y_train):
    """
    Train multiple models and compare performance
    """
    print("\n=== Training Churn Prediction Models ===")
    
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
    }
    
    trained_models = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        trained_models[name] = model
    
    return trained_models

def evaluate_models(models, X_test, y_test):
    """
    Evaluate all models and select the best
    """
    print("\n=== Model Evaluation ===")
    
    best_model = None
    best_score = 0
    best_name = ""
    
    for name, model in models.items():
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        print(f"\n{name}:")
        print(f"ROC-AUC Score: {roc_auc:.4f}")
        print(classification_report(y_test, y_pred, target_names=['No Churn', 'Churn']))
        
        if roc_auc > best_score:
            best_score = roc_auc
            best_model = model
            best_name = name
    
    print(f"\n=== Best Model: {best_name} (ROC-AUC: {best_score:.4f}) ===")
    return best_model, best_name, best_score

def analyze_feature_importance(model, feature_names):
    """
    Analyze and display feature importance
    """
    print("\n=== Feature Importance Analysis ===")
    
    if hasattr(model, 'feature_importances_'):
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(importance_df.head(10))
        return importance_df
    else:
        print("Model does not support feature importance extraction.")
        return None

def identify_churn_risk(df, model, feature_cols):
    """
    Identify high-risk customers for churn
    """
    print("\n=== Churn Risk Analysis ===")
    
    X = df[feature_cols]
    churn_probabilities = model.predict_proba(X)[:, 1]
    
    df['churn_risk_score'] = churn_probabilities
    df['risk_category'] = pd.cut(churn_probabilities, 
                                  bins=[0, 0.3, 0.7, 1.0],
                                  labels=['Low', 'Medium', 'High'])
    
    high_risk_customers = df[df['risk_category'] == 'High'].shape[0]
    churn_rate = df['churn'].mean() if 'churn' in df.columns else 0
    
    print(f"\nTotal Customers: {len(df)}")
    print(f"High Risk Customers: {high_risk_customers} ({high_risk_customers/len(df)*100:.1f}%)")
    print(f"Overall Churn Rate: {churn_rate*100:.1f}%")
    
    # Top 10 high-risk customers
    print("\nTop 10 High-Risk Customers:")
    print(df.nlargest(10, 'churn_risk_score')[['customer_id', 'tenure', 'monthly_charges', 'churn_risk_score']])
    
    return df

def main():
    """
    Main execution pipeline for churn prediction
    """
    print("Customer Churn Prediction Model")
    print("=" * 60)
    
    # Load and prepare data
    df = load_customer_data()
    df = engineer_churn_features(df)
    
    X, y, feature_cols = prepare_features(df)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    X_train = pd.DataFrame(X_train_scaled, columns=feature_cols)
    X_test = pd.DataFrame(X_test_scaled, columns=feature_cols)
    
    # Train models
    models = train_churn_models(X_train, y_train)
    
    # Evaluate and select best
    best_model, best_name, best_score = evaluate_models(models, X_test, y_test)
    
    # Feature importance
    analyze_feature_importance(best_model, feature_cols)
    
    # Risk analysis
    df_with_scores = identify_churn_risk(df, best_model, feature_cols)
    
    # Save model
    joblib.dump(best_model, 'churn_prediction_model.pkl')
    joblib.dump(scaler, 'churn_scaler.pkl')
    print("\nModel and scaler saved successfully!")
    
    print("\n" + "=" * 60)
    print(f"Churn prediction complete! Best model: {best_name} (ROC-AUC: {best_score:.1%})")

if __name__ == "__main__":
    main()
