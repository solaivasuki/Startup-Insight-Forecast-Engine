import streamlit as st
import pandas as pd
from forecast.predict_forecast import forecast_revenue
from charts import plot_revenue_chart
from rag.query_rag import get_structured_insight
from news.news_fetcher import fetch_live_news
import os 

# Suppress TF warning
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

st.set_page_config(page_title="Startup Forecast Dashboard", layout="wide")
st.markdown(
    """
    <style>
    body {
        background-color: #f5f7fa;
    }
    .stApp {
        background-color: #f5f7fa;
    }
    html, body, [class*="css"] {
        font-family: 'Segoe UI', sans-serif;
        color: #333333;
    }
    .element-container .stMetric {
        background: white;
        padding: 15px;
        border-radius: 12px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
    }
    </style>
    """,
    unsafe_allow_html=True
)
os.environ["STREAMLIT_WATCH_USE_POLLING"] = "true"


# Title
st.title("📈Startup Insight & Forecast Engine")

# Load data
data_path = os.path.join("data", "startups_2025.csv")
df = pd.read_csv(data_path)

# ✅ Company selector in sidebar
with st.sidebar:
    st.markdown("## 📊 Startup Explorer")

    startup_names = df["Startup Name"].dropna().unique().tolist()
    selected_startup = st.selectbox(
        "🔍 Search or Select a Startup", 
        ["-- Select a Startup --"] + startup_names, 
        index=0
    )

    if selected_startup == "-- Select a Startup --":
        st.warning("Please select a valid startup to continue.")
        st.stop()
with st.sidebar:
    st.sidebar.markdown("## 📰 Latest Startup News")
    news = fetch_live_news("startups")
    for item in news[:3]:
        st.sidebar.markdown(f"- [{item['title']}]({item['url']})")


# Tabs
tab1, tab2 = st.tabs(["📄 Company Details", "🧠 Ask AI Anything"])

with tab1:
        # Professional company name card
    st.markdown(f"""
        <div style="
            background-color: #ffffff;
            padding: 20px;
            border-left: 5px solid #4A90E2;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
            margin: 20px 0;
        ">
            <h3 style="color: #333333; margin: 0; font-family: 'Segoe UI', sans-serif;">
                🏢 {selected_startup}
            </h3>
        </div>
    """, unsafe_allow_html=True)






    # 🎯 Extract data for selected startup
    startup_data = df[df["Startup Name"] == selected_startup].iloc[0]

    # 🧮 Revenue metrics
    revenue_columns = ["Revenue_2019", "Revenue_2020", "Revenue_2021", "Revenue_2022", "Revenue_2023", "Revenue_2024"]
    revenue_values = startup_data[revenue_columns].values.astype(float)

    max_revenue = revenue_values.max()
    min_revenue = revenue_values.min()
    avg_revenue = revenue_values.mean()

    # 📊 Show key revenue metrics in cards
    st.markdown("### 💰 Key Revenue Metrics")
    col1, col2, col3 = st.columns(3)
    col1.metric("🔼 Max Revenue (Cr)", f"₹{max_revenue:,.2f}")
    col2.metric("🔽 Min Revenue (Cr)", f"₹{min_revenue:,.2f}")
    col3.metric("📊 Avg Revenue (Cr)", f"₹{avg_revenue:,.2f}")


    # Charts
    st.subheader("Revenue Performance & Forecast (2019–2026)")
    plot_revenue_chart(selected_startup, df)


    # 📘 Company Details Section
    st.markdown("---")
    st.subheader("🏢 Company Details")

    with st.container():
        col1, col2 = st.columns([1, 2])

        with col1:
            st.markdown("**Legal Name:**")
            st.markdown(startup_data["Startup Name"])
            st.markdown("**Operating Status:**")
            st.markdown(startup_data["CompanyStatus"])
            st.markdown("**Company Category:**")
            st.markdown(startup_data["CompanyCategory"])
            st.markdown("**Company Class:**")
            st.markdown(startup_data["CompanyClass"])

        with col2:
            st.markdown("**Authorized Capital:**")
            st.markdown(startup_data["AuthorizedCapital"])
            st.markdown("**Paidup Capital:**")
            st.markdown(startup_data["PaidupCapital"])
            st.markdown("**Contact Address:**")
            st.markdown(startup_data["Headquarters"])
            st.markdown("**Company Registration Date:**")
            st.markdown(startup_data["CompanyRegistrationdate_date"])


with tab2:
    st.subheader("💬 Ask Anything About Startups")

    st.markdown("Type your question below and get instant insights!")
    st.markdown("**Examples:**")
    st.markdown("- Top 5 companies")
    st.markdown("- Top industries in Chennai")
    st.markdown("- Most profitable company")
    st.markdown("- Who are the top AI startups in Bangalore?")

    question = st.text_input("📌 Your Question", placeholder="e.g., Most profitable AI startup in India")

    if question:
        get_structured_insight(question)


