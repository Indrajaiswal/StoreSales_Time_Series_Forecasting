import sys
import warnings

# Python 3.13 Streamlit patch (imghdr)
if sys.version_info >= (3, 13):
    import types
    warnings.warn(
        "⚠️ Python 3.13 detected. Streamlit uses 'imghdr' internally, "
        "which is removed in this Python version. "
        "Since your app doesn't use images, this patch prevents a crash."
    )
    sys.modules["imghdr"] = types.ModuleType("imghdr")
    
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import numpy as np

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="📊 Sales Forecast Dashboard",
    layout="wide",
    page_icon="📈"
)

# -----------------------------
# Load Data
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("data/daily_sales.csv", parse_dates=["date"])
    df.set_index("date", inplace=True)
    return df

data = load_data()

# -----------------------------
# Load Model
# -----------------------------
@st.cache_resource
def load_model():
    return joblib.load("arima_sales_model.pkl")

model = load_model()



# -----------------------------
# Helper Function
# -----------------------------
def format_currency(x):
    return f"₹{x:,.0f}"

# -----------------------------
# Title & Description
# -----------------------------
st.title("📊 Store Sales Forecasting Dashboard")
st.markdown("""
**Forecasting daily store sales using ARIMA.**  
Includes historical analysis, model performance, and future forecast with confidence intervals.
""")

# -----------------------------
# Tabs Layout
# -----------------------------
tab1, tab2, tab3 = st.tabs(["📈 Forecast", "📊 Model Comparison", "ℹ️ Model Summary"])


# -----------------------------
# KPI SECTION
# -----------------------------
col1, col2, col3 = st.columns(3)

col1.metric("Total Historical Sales", f"₹{data['sales'].sum():,.0f}")
col2.metric("Average Daily Sales", f"₹{data['sales'].mean():,.0f}")
col3.metric("Last Day Sales", f"₹{data['sales'].iloc[-1]:,.0f}")

# -----------------------------
# Historical Sales
# -----------------------------
st.subheader("📊 Historical Daily Sales Amount")
st.line_chart(data["sales"])


# =====================================================
# TAB 1: FORECAST
# =====================================================
with tab1:
    st.subheader("🔮 Sales Forecast")
    
    # Forecast horizon slider
    days = st.slider("Select forecast horizon (days)", 7, 90, 30)
    
    # Forecast calculation
    forecast_result = model.get_forecast(steps=days)
    forecast_mean = forecast_result.predicted_mean
    conf_int = forecast_result.conf_int()
    
    forecast_df = pd.DataFrame({
        "Date": pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=days),
        "Forecast": forecast_mean.values,
        "Lower CI": conf_int.iloc[:, 0].values,
        "Upper CI": conf_int.iloc[:, 1].values
    }).set_index("Date")
    
    # KPIs
    latest_sales = data["sales"].iloc[-1]
    avg_forecast = forecast_df["Forecast"].mean()
    growth = ((avg_forecast - latest_sales) / latest_sales) * 100
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Latest Sales", format_currency(latest_sales))
    c2.metric("Avg Forecast", format_currency(avg_forecast))
    c3.metric("Expected Change", f"{growth:.2f}%")
    
    # Plot forecast
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(data.index[-120:], data["sales"].iloc[-120:], label="Historical")
    ax.plot(forecast_df.index, forecast_df["Forecast"], label="Forecast", marker="o")
    ax.fill_between(
        forecast_df.index,
        forecast_df["Lower CI"],
        forecast_df["Upper CI"],
        color='skyblue',
        alpha=0.3,
        label="Confidence Interval"
    )
    ax.set_title("Daily Sales Forecast with Confidence Intervals")
    ax.set_ylabel("Sales Amount")
    ax.legend()
    st.pyplot(fig)
    
    # Forecast table
    st.subheader("📄 Forecast Table")
    forecast_display = forecast_df.applymap(format_currency)
    st.dataframe(forecast_display)
    
    # -----------------------------
    # Highlight Top 5 Forecasted Days
    # -----------------------------
    st.subheader("🏆 Top 5 Forecasted Sales Days")
    top5 = forecast_df["Forecast"].sort_values(ascending=False).head(5)
    top5_display = top5.apply(format_currency)
    st.table(top5_display)
    
    # Download forecast
    st.download_button(
        "⬇️ Download Forecast CSV",
        data=forecast_df.to_csv().encode("utf-8"),
        file_name="sales_forecast.csv",
        mime="text/csv"
    )

# =====================================================
# TAB 2: MODEL PERFORMANCE
# =====================================================
with tab2:
    st.subheader("📊 Model Comparison")
    
    performance_df = pd.DataFrame({
        "Model": ["ARIMA", "SARIMAX", "Prophet", "Prophet + Holidays"],
        "MAE": [26505.89, 69032.02, 37628.94, 37628.94],
        "RMSE": [43422.49, 87154.60, 50568.55, 50568.55]
    })
    
    st.dataframe(performance_df)
    
    st.markdown("""
    **Key Observations**
    - ARIMA has the lowest MAE and RMSE
    - SARIMAX underperformed despite exogenous variables
    - Holidays did not improve Prophet accuracy
    """)
    
    # Bar chart for MAE
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    ax2.bar(performance_df["Model"], performance_df["MAE"], color="teal")
    ax2.set_title("MAE Comparison Across Models")
    ax2.set_ylabel("MAE")
    st.pyplot(fig2)

# =====================================================
# TAB 3: MODEL SUMMARY
# =====================================================
with tab3:
    st.subheader("ℹ️ Model Summary & Conclusion")
    
    st.markdown("""
    ### ✔ Final Model Choice: **ARIMA**
    
    **Reasons**
    - Lowest MAE and RMSE across all models
    - Strong generalization in cross-validation
    - Simple, interpretable, production-ready
    
    **Why not SARIMAX?**
    - Promotions did not significantly improve predictions
    - Higher error and poor cross-validation stability
    
    **Why not Prophet?**
    - Useful for trend visualization
    - Lower accuracy than ARIMA
    - Holiday effects were insignificant
    
    ### 🎯 Business Impact
    - Reliable short-term sales forecasting
    - Supports inventory & demand planning
    - Easy deployment and monitoring
    """)



# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.caption("Developed as an end-to-end Time Series Forecasting Project")
