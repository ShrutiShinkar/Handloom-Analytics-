# src/app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter

# --- Page Configuration ---
st.set_page_config(page_title="Indian Handloom Analytics", layout="wide")

# --- Caching for Performance ---
@st.cache_data
def load_and_process_data(file_path):
    """Loads and preprocesses data for visualizations."""
    df = pd.read_csv(file_path)
    numerical_columns = df.select_dtypes(include='float64').columns
    for col in numerical_columns:
        df[col].fillna(df[col].median(), inplace=True)
    df['loan'] = df.iloc[:, 64:72].sum(axis=1)
    df['Sales'] = df.iloc[:, 86:93].sum(axis=1)
    df['Debt'] = df.iloc[:, 72:75].sum(axis=1)
    df['total_weavers'] = df['male_weavers'] + df['female_weavers']
    return df

# --- Load Model and Data ---
try:
    model = joblib.load('src/income_model.pkl')
    encoder = joblib.load('src/income_encoder.pkl')
    df_plots = load_and_process_data('data/handloom sales.csv')
except FileNotFoundError:
    st.error("Model not found. Please run `python -m src.model_training` first from your terminal.")
    st.stop()

# --- Main Page Layout ---
st.title("📊 Predictive Analytics for the Indian Handloom Industry")
st.markdown("An interactive dashboard to predict the financial standing of handloom households and explore the complete dataset analysis from the original notebook.")

# --- Using Tabs for Organization ---
tab1, tab2, tab3, tab4 = st.tabs([
    "🔮 **Income Predictor**",
    "📈 **Sales Analysis**",
    "📄 **Debt & Demographics**",
    "🧠 **Correlation Analysis**"
])

# --- TAB 1: PREDICTOR ---
with tab1:
    st.header("Predict a Household's Income Category")

    with st.sidebar:
        st.header("Input Features")
        total_hh = st.slider('Total Household Members', 1, 20, 5, key="pred_hh")
        male_weavers = st.slider('Number of Male Weavers', 0, 10, 1, key="pred_male")
        female_weavers = st.slider('Number of Female Weavers', 0, 10, 1, key="pred_female")
        loan = st.slider('Total Loan Amount (INR)', 0, 500000, 50000, key="pred_loan")
        sales = st.slider('Total Annual Sales (INR)', 0, 1000000, 100000, key="pred_sales")

    # --- Feature Engineering for Prediction ---
    total_weavers = male_weavers + female_weavers
    weavers_per_hh = total_weavers / (total_hh + 1e-6)
    loan_to_sales_ratio = loan / (sales + 1e-6)

    features_df = pd.DataFrame([[
        total_hh, male_weavers, female_weavers, loan, sales,
        total_weavers, weavers_per_hh, loan_to_sales_ratio
    ]], columns=[
        'total_hh', 'male_weavers', 'female_weavers', 'loan', 'Sales',
        'total_weavers', 'weavers_per_hh', 'loan_to_sales_ratio'
    ])

    features_df['loan'] = np.log1p(features_df['loan'])
    features_df['Sales'] = np.log1p(features_df['Sales'])
    features_df['loan_to_sales_ratio'] = np.log1p(features_df['loan_to_sales_ratio'])

    prediction_encoded = model.predict(features_df)
    prediction_text = encoder.inverse_transform(prediction_encoded)[0]
    probabilities = model.predict_proba(features_df)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Prediction")
        st.metric("Predicted Income Category", value=prediction_text)
    with col2:
        st.subheader("Model Confidence")
        prob_df = pd.DataFrame(probabilities, columns=encoder.classes_, index=['Probability'])
        st.bar_chart(prob_df.T)

# --- TAB 2: SALES ANALYSIS ---
with tab2:
    st.header("Analysis of Sales, Materials, and Markets")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Total Sales vs Different Dye Types")
        dye_sales = df_plots.iloc[:, 82:86].sum()
        fig, ax = plt.subplots()
        ax.bar(dye_sales.index, dye_sales.values, color=['#4C72B0', '#55A868', '#C44E52', '#8172B2'])
        plt.xticks(rotation=45, ha='right')
        ax.set_ylabel("Total Sales")
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, loc: "{:,}".format(int(x))))
        st.pyplot(fig)

    with col2:
        st.subheader("Sales vs Residence Type")
        fig, ax = plt.subplots()
        residence_sales = df_plots.groupby('type_residence')['Sales'].sum()
        sns.lineplot(x=residence_sales.index, y=residence_sales.values, ax=ax, marker='o', color='#DD8452')
        ax.set_ylabel("Total Sales")
        ax.set_xlabel("Residence Type")
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, loc: "{:,}".format(int(x))))
        st.pyplot(fig)
        
    st.subheader("Total Sales vs Different Yarn Types")
    yarn_sales = df_plots.iloc[:, 48:64].sum()
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(yarn_sales.index, yarn_sales.values, color=plt.cm.plasma(np.linspace(0, 1, len(yarn_sales))))
    plt.xticks(rotation=75, ha='right')
    ax.set_ylabel("Total Sales")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, loc: "{:,}".format(int(x))))
    st.pyplot(fig)

    st.subheader("Sales Distribution by Dye Type Across Different Markets")
    sales_columns = df_plots.columns[86:93]
    dye_columns = df_plots.columns[82:86]

    for sale_column in sales_columns:
        fig, ax = plt.subplots()
        for dye_column in dye_columns:
            filtered_data = df_plots[df_plots[dye_column] == 1][sale_column]
            sns.histplot(filtered_data, bins=10, kde=True, alpha=0.5, label=dye_column.replace('_dye', ''), ax=ax)
        
        ax.set_title(f"Histogram of {sale_column.replace('_', ' ').title()}")
        ax.set_xlabel("Sales in Market")
        ax.legend(title="Dye Type")
        st.pyplot(fig)

# --- TAB 3: DEBT & DEMOGRAPHICS ---
with tab3:
    st.header("Analysis of Debt, Income, and Demographics")
    
    # --- BUG FIX: Explicitly define income columns to prevent errors ---
    income_columns = [
        'income_less_5000_hl_related', 'income_btw_5001_10000_hl_related',
        'income_btw_10001_15000_hl_related', 'income_btw_15001_20000_hl_related',
        'income_btw_20001_25000_hl_related', 'income_btw_25001_50000_hl_related',
        'income_btw_50001_100000_hl_related', 'income_above_100000_hl_related'
    ]
    labels = ['< ₹5k', '₹5-10k', '₹10-15k', '₹15-20k', '₹20-25k', '₹25-50k', '₹50k-1L', '> ₹1L']

    st.subheader("Distribution of Income Categories")
    total_households = [df_plots[col].sum() for col in income_columns]
    fig, ax = plt.subplots()
    wedges, _, _ = ax.pie(total_households, autopct='%1.1f%%', startangle=140, textprops=dict(color="w"), pctdistance=0.85, colors=plt.cm.coolwarm(np.linspace(0, 1, len(labels))))
    ax.legend(wedges, labels, title="Income Categories", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
    st.pyplot(fig)

    st.subheader("Loan Distribution by Income Level")
    all_loans_for_plot = []
    for col in income_columns:
        loans_in_category = df_plots.loc[df_plots[col] == 1, 'loan']
        all_loans_for_plot.append(loans_in_category)
        
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.boxplot(all_loans_for_plot, vert=True, patch_artist=True)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_xlabel("Income Category")
    ax.set_ylabel("Loan Amount")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, loc: "{:,}".format(int(x))))
    st.pyplot(fig)

# --- TAB 4: CORRELATION ANALYSIS ---
with tab4:
    st.header("Correlation Analysis")
    st.subheader("Correlation Heatmaps of Key Numeric Variables")
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Debt, Loan, Sales & Demographics**")
        corr_features = ['Debt', 'loan', 'Sales', 'total_hh', 'total_weavers']
        corr_matrix = df_plots[corr_features].corr(method='spearman')
        fig, ax = plt.subplots()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
        st.pyplot(fig)

    with col2:
        st.write("**Loan & Debt vs Income Categories**")
        income_corr_df = df_plots.loc[:, list(income_columns) + ['loan', 'Debt']].corr().loc[income_columns, ['loan', 'Debt']]
        fig, ax = plt.subplots(figsize=(6, 8))
        sns.heatmap(income_corr_df, annot=True, cmap='viridis', fmt=".2f", ax=ax, yticklabels=labels)
        plt.tick_params(axis='y', labelsize=10)
        st.pyplot(fig)