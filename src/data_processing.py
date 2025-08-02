# src/data_processing.py
import pandas as pd
import numpy as np

def load_and_process_data(file_path):
    """
    Loads and performs all preprocessing and feature engineering steps
    from the notebook to be used consistently across the project.
    """
    df = pd.read_csv(file_path)

    # Fill Missing Values
    numerical_columns = df.select_dtypes(include='float64').columns
    for col in numerical_columns:
        df[col].fillna(df[col].median(), inplace=True)

    # Merge Columns using column names for robustness
    loan_cols = [
        'loan_from_cooperative_societies', 'loan_from_commercial_banks', 'loan_from_friends_or_relatives',
        'loan_from_govt', 'loan_from_master_weaver', 'loan_from_money_lender',
        'loan_from_other_sources', 'loan_from_shg'
    ]
    df['loan'] = df[loan_cols].sum(axis=1)

    sales_cols = [
        'sales_local_market', 'sales_master_weaver_group', 'sales_cooperative_society',
        'sales_organized_fairs_or_exhibition', 'sales_export', 'sales_ecommerce', 'sales_other_sources'
    ]
    df['Sales'] = df[sales_cols].sum(axis=1)
    
    debt_cols = [
        'debt_due_to_hl', 'debt_due_to_other_purposes', 'debt_due_to_hl_and_other_purposes'
    ]
    df['Debt'] = df[debt_cols].sum(axis=1)
    
    # Feature Engineering for the model
    df['total_weavers'] = df['male_weavers'] + df['female_weavers']
    df['weavers_per_hh'] = df['total_weavers'] / (df['total_hh'] + 1e-6)
    df['loan_to_sales_ratio'] = df['loan'] / (df['Sales'] + 1e-6)
    
    return df