# dashboard.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Railway Demand Dashboard",
    page_icon="🚆",
    layout="wide"
)

# --- LOAD DATA AND MODELS (CACHE TO AVOID RELOADING) ---
@st.cache_data
# REPLACE your old load_data function with this one

@st.cache_data
def load_data():
    """Loads the main dataframe, the aggregated data, the model, and encoders."""
    df = pd.read_csv("Quota_wise_demand.csv")

    # --- THIS NEW LINE FIXES THE ERROR ---
    df.columns = df.columns.str.strip(" '\"")
    # -------------------------------------
    
    df['journey_date'] = pd.to_datetime(df['journey_date'])
    df['weekday'] = df['journey_date'].dt.day_name()
    
    # Clean string columns
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str).str.strip().str.replace("'", "")
        
    agg_df = pd.read_csv("aggregated_demand_data.csv")
    
    # Load model and encoders
    try:
        model = joblib.load("quota_xgb_model.pkl")
        le_cls = joblib.load("label_encoder_cls.pkl")
        le_quota = joblib.load("label_encoder_quota.pkl")
    except FileNotFoundError:
        st.error("Model files not found. Please run the main training script first.")
        model, le_cls, le_quota = None, None, None
        
    return df, agg_df, model, le_cls, le_quota

df, agg_df, model, le_cls, le_quota = load_data()


# --- PREDICTION FUNCTION ---
# COPY THIS ENTIRE BLOCK

def predict_single_day(model, le_cls, le_quota, future_date, pred_cls, pred_quota, df):
    """Predict demand for a single day with proper feature engineering."""
    
    # Filter historical data for the selected class and quota
    hist = df[(df['trvl_cls_desc'] == pred_cls) & 
              (df['quota_enc'] == pred_quota)].copy()
    
    if hist.empty:
        return None
    
    hist = hist.sort_values('journey_date')
    
    # Calculate ALL lag features (must match training exactly)
    hist['lag1'] = hist['no_of_pax'].shift(1)
    hist['lag2'] = hist['no_of_pax'].shift(2)
    hist['lag3'] = hist['no_of_pax'].shift(3)
    hist['lag7'] = hist['no_of_pax'].shift(7)
    hist['lag14'] = hist['no_of_pax'].shift(14)
    hist['lag30'] = hist['no_of_pax'].shift(30)
    
    # Calculate rolling statistics
    hist['rolling_7_mean'] = hist['no_of_pax'].rolling(window=7).mean()
    hist['rolling_7_std'] = hist['no_of_pax'].rolling(window=7).std()
    hist['rolling_30_mean'] = hist['no_of_pax'].rolling(window=30).mean()
    
    # Calculate EWMA
    hist['ewma_14'] = hist['no_of_pax'].ewm(span=14).mean()
    
    # Extract temporal features
    hist['dayofweek'] = hist['journey_date'].dt.dayofweek
    hist['month'] = hist['journey_date'].dt.month
    hist['is_weekend'] = hist['journey_date'].dt.dayofweek.isin([5, 6]).astype(int)
    hist['quarter'] = hist['journey_date'].dt.quarter
    
    # Get the most recent complete row as the base
    last_row = hist.dropna().iloc[-1]
    
    # Create prediction dataframe with future date features
    future_df = pd.DataFrame({
        'trvl_cls_desc_enc': [le_cls.transform([pred_cls])[0]],
        'quota_enc_enc': [le_quota.transform([pred_quota])[0]],
        'dayofweek': [pd.to_datetime(future_date).dayofweek],
        'month': [pd.to_datetime(future_date).month],
        'is_weekend': [1 if pd.to_datetime(future_date).dayofweek in [5, 6] else 0],
        'quarter': [pd.to_datetime(future_date).quarter],
        'lag1': [last_row['lag1']],
        'lag2': [last_row['lag2']],
        'lag3': [last_row['lag3']],
        'lag7': [last_row['lag7']],
        'lag14': [last_row['lag14']],
        'lag30': [last_row['lag30']],
        'rolling_7_mean': [last_row['rolling_7_mean']],
        'rolling_7_std': [last_row['rolling_7_std']],
        'rolling_30_mean': [last_row['rolling_30_mean']],
        'ewma_14': [last_row['ewma_14']]
    })
    
    # Make prediction
    prediction = model.predict(future_df)[0]
    
    return max(0, round(prediction))

# --- SIDEBAR FOR NAVIGATION ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Exploratory Dashboard", "Demand Forecaster"])


# =====================================================================================
# --- PAGE 1: EXPLORATORY DASHBOARD ---
# =====================================================================================
if page == "Exploratory Dashboard":
    st.title("🚆 Exploratory Analysis of Railway Demand")
    st.markdown("Use the filters in the sidebar to explore the dataset.")

    # --- SIDEBAR FILTERS FOR EDA ---
    st.sidebar.header("Dashboard Filters")
    
    # Date Range Filter
    min_date = df['journey_date'].min().date()
    max_date = df['journey_date'].max().date()
    start_date, end_date = st.sidebar.date_input(
        "Select Date Range", [min_date, max_date], min_value=min_date, max_value=max_date
    )

    # Class and Quota Filters
    selected_classes = st.sidebar.multiselect("Select Travel Class", options=sorted(df['cls'].unique()), default=sorted(df['cls'].unique()))
    selected_quotas = st.sidebar.multiselect("Select Quota", options=sorted(df['QUOTA_CODE'].unique()), default=sorted(df['QUOTA_CODE'].unique()))

    # Filter dataframe based on selections
    filtered_df = df[
        (df['journey_date'].dt.date >= start_date) &
        (df['journey_date'].dt.date <= end_date) &
        (df['cls'].isin(selected_classes)) &
        (df['QUOTA_CODE'].isin(selected_quotas))
    ]

    # --- KPIs ---
    st.header("Key Metrics")
    total_passengers = int(filtered_df['PSGN'].sum())
    avg_passengers = round(filtered_df['PSGN'].mean(), 2)

    col1, col2 = st.columns(2)
    col1.metric("Total Passengers", f"{total_passengers:,}")
    col2.metric("Average Passengers per Booking", avg_passengers)

    # --- CHARTS ---
    st.header("Visualizations")
    
    # Demand Over Time
    time_series = filtered_df.groupby('journey_date')['PSGN'].sum().reset_index()
    fig_time = px.line(time_series, x='journey_date', y='PSGN', title='Passenger Demand Over Time')
    st.plotly_chart(fig_time, use_container_width=True)

    # Demand by Class and Quota
    col3, col4 = st.columns(2)
    class_demand = filtered_df.groupby('cls')['PSGN'].sum().sort_values(ascending=False).reset_index()
    fig_class = px.bar(class_demand, x='cls', y='PSGN', title='Demand by Travel Class', color='cls')
    col3.plotly_chart(fig_class, use_container_width=True)

    quota_demand = filtered_df.groupby('QUOTA_CODE')['PSGN'].sum().sort_values(ascending=False).reset_index().head(10)
    fig_quota = px.bar(quota_demand, x='QUOTA_CODE', y='PSGN', title='Top 10 Quotas by Demand', color='QUOTA_CODE')
    col4.plotly_chart(fig_quota, use_container_width=True)

# =====================================================================================
# --- PAGE 2: DEMAND FORECASTER ---
# =====================================================================================
elif page == "Demand Forecaster":
    st.title("🔮 Predict Future Passenger Demand")
    st.markdown("Select a date, class, and quota to get a demand forecast.")
    
    # --- USER INPUTS FOR PREDICTION ---
    col1, col2, col3 = st.columns(3)
    
    with col1:
        future_date = st.date_input("Select a Future Date", 
                                     value=pd.to_datetime('today') + pd.Timedelta(days=7))
    with col2:
        pred_cls = st.selectbox("Select Travel Class", 
                                options=sorted(le_cls.classes_))
    with col3:
        pred_quota = st.selectbox("Select Quota", 
                                  options=sorted(le_quota.classes_))
    
    # --- PREDICTION BUTTON AND OUTPUT ---
    if st.button("Forecast Demand", type="primary"):
        features_list = ['dayofweek', 'month', 'is_weekend', 'clsenc', 'quotaenc', 
                        'lag1', 'lag2', 'lag3', 'lag7', 'lag14', 'lag30', 
                        'rolling_7_mean', 'dayofyear', 'weekofyear', 'quarter', 
                        'rolling_30_mean', 'ewma_14']
        
        with st.spinner("Forecasting..."):
            # CORRECTED: Pass all 7 required arguments
            prediction = predict_single_day(
                future_date,    # date
                pred_cls,       # cls
                pred_quota,     # quota
                features_list,  # features_list
                model,          # model (from load_data)
                le_cls,         # le_cls (from load_data)
                le_quota,       # le_quota (from load_data)
                df              # df (from load_data)
            )
        
        st.subheader("📊 Forecast Result")
        if isinstance(prediction, str):
            st.error(prediction)
        else:
            st.metric(
                label=f"Predicted Passengers for {pred_cls} - {pred_quota}",
                value=f"{prediction} passengers"
            )


# Add this to the end of your main training script
agg_df.to_csv("aggregated_demand_data.csv", index=False)
print("\nAggregated data saved for dashboard use.")