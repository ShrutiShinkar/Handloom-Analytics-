# src/app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# --- Page Configuration ---
st.set_page_config(
    page_title="Indian Handloom Analytics",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Caching for Performance ---
@st.cache_data
def load_data(file_path):
    """Loads and preprocesses data, cached for performance."""
    df = pd.read_csv(file_path)
    # Perform minimal processing needed for visualizations here
    sales_cols = [
        'sales_local_market', 'sales_master_weaver_group', 'sales_cooperative_society',
        'sales_organized_fairs_or_exhibition', 'sales_export', 'sales_ecommerce', 'sales_other_sources'
    ]
    df['total_sales'] = df[sales_cols].sum(axis=1)
    return df

# --- Load Model and Data ---
try:
    model = joblib.load('src/income_model.pkl')
    encoder = joblib.load('src/income_encoder.pkl')
    df_raw = load_data('data/handloom sales.csv')
except FileNotFoundError:
    st.error("Model not found. Please run `python -m src.model_training` first from your terminal.")
    st.stop()

# --- Sidebar for User Input ---
with st.sidebar:
    st.header("Predict Household Income")
    st.write("Adjust the sliders to see the predicted income category for a handloom household.")

    total_hh = st.slider('Total Household Members', 1, 20, 5)
    male_weavers = st.slider('Number of Male Weavers', 0, 10, 1)
    female_weavers = st.slider('Number of Female Weavers', 0, 10, 1)
    loan = st.slider('Total Loan Amount (INR)', 0, 500000, 50000)
    sales = st.slider('Total Annual Sales (INR)', 0, 1000000, 100000)
    
    predict_button = st.button('Predict Income Category', type="primary")

# --- Main Page Layout ---
st.title("📊 Predictive Analytics for the Indian Handloom Industry")
st.markdown("This application uses a Random Forest model to predict the financial standing of handloom households and provides key insights from the national handloom census data.")

# --- Prediction Display ---
st.header("🔮 Income Prediction")
col1, col2 = st.columns([1, 2])

with col1:
    st.write("**Your Selections:**")
    input_data = {
        'Total HH': total_hh, 'Male Weavers': male_weavers,
        'Female Weavers': female_weavers, 'Loan (INR)': f"₹{loan:,}", 'Sales (INR)': f"₹{sales:,}"
    }
    st.json(input_data)

with col2:
    if predict_button:
        # Prepare input for the model
        features_df = pd.DataFrame([[total_hh, male_weavers, female_weavers, loan, sales]], 
                                   columns=['total_hh', 'male_weavers', 'female_weavers', 'loan', 'Sales'])
        # Apply the same log transformation as in training
        features_df['loan'] = np.log1p(features_df['loan'])
        features_df['Sales'] = np.log1p(features_df['Sales'])

        # Get prediction and probabilities
        prediction_encoded = model.predict(features_df)
        prediction_text = encoder.inverse_transform(prediction_encoded)[0]
        probabilities = model.predict_proba(features_df)

        st.metric(label="Predicted Income Category", value=prediction_text)

        st.write("**Model Confidence:**")
        prob_df = pd.DataFrame(probabilities, columns=encoder.classes_, index=['Probability'])
        st.bar_chart(prob_df.T)
    else:
        st.info("Adjust the sliders on the left and click 'Predict' to see the result.")

st.markdown("---")

# --- Data Visualizations ---
st.header("📈 Visual Insights")

# Two columns for side-by-side charts
col3, col4 = st.columns(2)

with col3:
    st.subheader("Distribution of Income Categories")
    income_columns = [
        'income_less_5000_hl_related', 'income_btw_5001_10000_hl_related',
        'income_btw_10001_15000_hl_related', 'income_btw_15001_20000_hl_related',
        'income_btw_20001_25000_hl_related', 'income_btw_25001_50000_hl_related',
        'income_btw_50001_100000_hl_related', 'income_above_100000_hl_related'
    ]
    total_households = [df_raw[col].sum() for col in income_columns]
    labels = ['< ₹5k', '₹5-10k', '₹10-15k', '₹15-20k', '₹20-25k', '₹25-50k', '₹50k-1L', '> ₹1L']

    fig1, ax1 = plt.subplots()
    wedges, texts, autotexts = ax1.pie(total_households, autopct='%1.1f%%', startangle=140, textprops=dict(color="w"))
    ax1.axis('equal')
    ax1.legend(wedges, labels, title="Income Categories", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
    st.pyplot(fig1)

with col4:
    st.subheader("Total Sales by Dye Type")
    dye_sales = df_raw.iloc[:, 82:86].sum()
    fig2, ax2 = plt.subplots()
    sns.barplot(x=dye_sales.index, y=dye_sales.values, ax=ax2)
    ax2.set_ylabel("Total Sales (in Crores)")
    ax2.set_xlabel("Dye Types")
    ax2.tick_params(axis='x', rotation=30)
    st.pyplot(fig2)

# --- Raw Data ---
with st.expander("View Raw Data Sample"):
    st.dataframe(df_raw.head())