import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
from data_processing import load_data, preprocess_data

def train_and_save_model():
    """Trains a more powerful income prediction model and saves it."""
    # Load and preprocess data
    df_raw = load_data('data/handloom sales.csv')
    df_processed = preprocess_data(df_raw.copy())

    # Create the target variable 'Income_Category'
    income_cols = [
        'income_less_5000_hl_related', 'income_btw_5001_10000_hl_related',
        'income_btw_10001_15000_hl_related', 'income_btw_15001_20000_hl_related',
        'income_btw_20001_25000_hl_related', 'income_btw_25001_50000_hl_related',
        'income_btw_50001_100000_hl_related', 'income_above_100000_hl_related'
    ]
    df_processed['Total_Income_Proxy'] = df_raw[income_cols].sum(axis=1)
    df_processed['Income_Category'] = pd.qcut(
        df_processed['Total_Income_Proxy'], 3, labels=["Low", "Medium", "High"]
    )

    # Select Features and Target
    features = ['total_hh', 'male_weavers', 'female_weavers', 'loan', 'Sales']
    target = 'Income_Category'

    X = df_processed[features]
    y = df_processed[target]

    X['loan'] = np.log1p(X['loan'])
    X['Sales'] = np.log1p(X['Sales'])

    # Encode the target variable
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Split data and train the new RandomForest model
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
    
    # Use RandomForestClassifier instead of DecisionTreeClassifier
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    print(f"Model trained successfully with accuracy: {model.score(X_test, y_test):.2f}")

    # Save the model and encoder
    joblib.dump(model, 'src/income_model.pkl')
    joblib.dump(le, 'src/income_encoder.pkl')
    print("Model ('income_model.pkl') and encoder ('income_encoder.pkl') have been saved in the 'src' folder.")

if __name__ == '__main__':
    train_and_save_model()