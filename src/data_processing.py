import pandas as pd

def load_data(file_path):
    """Loads data from the CSV file."""
    return pd.read_csv(file_path)

def preprocess_data(df):
    """Cleans, preprocesses, and feature-engineers the handloom data."""
    # Fill Missing Numerical Values with Median
    numerical_columns = df.select_dtypes(include='float64').columns
    for col in numerical_columns:
        median = df[col].median()
        df[col].fillna(median, inplace=True)

    # Merge Loan Columns
    loan_cols = [
        'loan_from_cooperative_societies', 'loan_from_commercial_banks',
        'loan_from_friends_or_relatives', 'loan_from_govt',
        'loan_from_master_weaver', 'loan_from_money_lender',
        'loan_from_other_sources', 'loan_from_shg'
    ]
    df['loan'] = df[loan_cols].sum(axis=1)

    # Merge Sales Columns
    sales_cols = [
        'sales_local_market', 'sales_master_weaver_group',
        'sales_cooperative_society', 'sales_organized_fairs_or_exhibition',
        'sales_export', 'sales_ecommerce', 'sales_other_sources'
    ]
    df['Sales'] = df[sales_cols].sum(axis=1)
    
    # Merge Debt Columns
    debt_cols = [
        'debt_due_to_hl', 'debt_due_to_other_purposes', 
        'debt_due_to_hl_and_other_purposes'
    ]
    df['Debt'] = df[debt_cols].sum(axis=1)

    return df