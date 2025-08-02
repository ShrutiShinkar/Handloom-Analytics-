# src/model_training.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
from data_processing import load_and_process_data

def train_and_save_model():
    """Trains a highly optimized income prediction model using GridSearchCV."""
    df_processed = load_and_process_data('data/handloom sales.csv')

    # Create the 'Income_Category' target variable
    income_cols = [
        'income_less_5000_hl_related', 'income_btw_5001_10000_hl_related',
        'income_btw_10001_15000_hl_related', 'income_btw_15001_20000_hl_related',
        'income_btw_20001_25000_hl_related', 'income_btw_25001_50000_hl_related',
        'income_btw_50001_100000_hl_related', 'income_above_100000_hl_related'
    ]
    df_processed['Total_Income_Proxy'] = df_processed[income_cols].sum(axis=1)
    df_processed['Income_Category'] = pd.qcut(
        df_processed['Total_Income_Proxy'], 3, labels=["Low", "Medium", "High"]
    )

    # Select all features the model will use
    features = [
        'total_hh', 'male_weavers', 'female_weavers', 'loan', 'Sales',
        'total_weavers', 'weavers_per_hh', 'loan_to_sales_ratio'
    ]
    target = 'Income_Category'

    X = df_processed[features]
    y = df_processed[target]
    
    # Apply log transformation to handle large, skewed numbers
    X['loan'] = np.log1p(X['loan'])
    X['Sales'] = np.log1p(X['Sales'])
    X['loan_to_sales_ratio'] = np.log1p(X['loan_to_sales_ratio'])

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
    
    param_grid = {'n_estimators': [100, 200], 'max_depth': [10, 20], 'min_samples_leaf': [1, 2]}
    rf = RandomForestClassifier(random_state=42, n_jobs=-1)
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=1, scoring='accuracy')

    print("Starting hyperparameter tuning...")
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    print(f"Best model trained with accuracy: {best_model.score(X_test, y_test):.2f}")

    joblib.dump(best_model, 'src/income_model.pkl')
    joblib.dump(le, 'src/income_encoder.pkl')
    print("Best model and encoder have been saved.")

if __name__ == '__main__':
    train_and_save_model()