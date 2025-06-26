import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import os
import pandas as pd
import streamlit as st

# === Load embedding model ===
model = SentenceTransformer('all-MiniLM-L6-v2')

# === Load FAISS and pkl ===
base_dir = os.path.dirname(os.path.abspath(__file__))
faiss_file = os.path.join(base_dir, "startup_index.faiss")
texts_file = os.path.join(base_dir, "startup_texts.pkl")

#print("Loading FAISS index from:", faiss_file)
#print("Loading texts from:", texts_file)

index = faiss.read_index(faiss_file)
with open(texts_file, "rb") as f:
    texts = pickle.load(f)

# === Semantic Search (Top-K startup entries) ===
def query_startups(query, top_k=5):
    query_embedding = model.encode([query])
    query_embedding = np.array(query_embedding).astype("float32")
    distances, indices = index.search(query_embedding, top_k)
    return [texts[i] for i in indices[0]]

# === Parse .pkl text into structured DataFrame ===
def parse_texts_to_df(texts):
    data = []
    for txt in texts:
        try:
            name = txt.split(" in ")[0]
            industry = txt.split(" in ")[1].split(" at ")[0]
            hq = txt.split(" at ")[1].split(" with ")[0]
            revenue = txt.split(" with Rs.")[1].split(" Cr")[0]
            revenue_val = float(revenue.replace(",", "").strip())
            data.append({
                "Startup Name": name.strip(),
                "Industry": industry.strip(),
                "Headquarters": hq.strip(),
                "Total Revenue": revenue_val
            })
        except Exception:
            continue
    return pd.DataFrame(data)

df = parse_texts_to_df(texts)

# === Load FLAN-T5 LLM ===
generator = pipeline("text2text-generation", model="google/flan-t5-base")

def format_revenue(value):
    return f"₹{value:,.2f} Cr"

def display_card(title, subtitle, revenue, location, index=None):
    bg_color = "#f5f5f5" if index % 2 == 0 else "#ffffff"
    st.markdown(
        f"""
        <div style="padding: 1rem; border-radius: 10px; background-color: {bg_color}; margin-bottom: 1rem; box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
            <h4 style="margin-bottom: 0.2rem;">{title}</h4>
            <small style="color: #555;">{subtitle}</small><br>
            <b>Revenue:</b> {revenue}<br>
            <b>Location:</b> {location}
        </div>
        """, unsafe_allow_html=True
    )

def get_structured_insight(query):
    query_lower = query.lower()

    # 🎯 Case 1: Top 5 companies overall
    if "top 5 companies" in query_lower:
        top = df.sort_values(by="Total Revenue", ascending=False).head(5).reset_index(drop=True)
        st.markdown("## 🏆 Top 5 Companies by Revenue")
        for i, row in top.iterrows():
            display_card(row["Startup Name"], row["Industry"], format_revenue(row["Total Revenue"]), row["Headquarters"], i)
        return

    # 🎯 Case 2: Top industries in [location]
    elif "top" in query_lower and "industries" in query_lower:
        # Attempt to extract a city
        for city in ["Chennai", "Coimbatore", "Madurai", "Salem", "Tiruchirapalli", "Erode"]:
            if city.lower() in query_lower:
                filtered = df[df["Headquarters"].str.contains(city, case=False, na=False)]
                if filtered.empty:
                    st.warning(f"No data found for startups in {city}.")
                    return
                industry_revenue = (
                    filtered.groupby("Industry")["Total Revenue"]
                    .sum()
                    .sort_values(ascending=False)
                    .head(5)
                )
                st.markdown(f"### 🏭 Top Industries in {city}")
                for i, (industry, rev) in enumerate(industry_revenue.items(), 1):
                    st.markdown(f"{i}. **{industry}** – ₹{rev:,.2f} Cr")
                st.bar_chart(industry_revenue)
                return

    # 🎯 Case 3: Most profitable company
    elif "most profitable" in query_lower or "highest revenue" in query_lower:
        top_company = df.sort_values(by="Total Revenue", ascending=False).iloc[0]
        st.markdown("## 💰 Most Profitable Company")
        display_card(top_company["Startup Name"], top_company["Industry"], format_revenue(top_company["Total Revenue"]), top_company["Headquarters"], 0)
        return

    # 🧠 Case 4: Use semantic + LLM fallback
    else:
        matched_texts = list(set(query_startups(query, top_k=10)))  # remove duplicates
        if not matched_texts:
            st.warning("No matching data found.")
            return

        # Extract info into structured form
        fallback_data = []
        for txt in matched_texts:
            try:
                name = txt.split(" in ")[0]
                industry = txt.split(" in ")[1].split(" at ")[0]
                hq = txt.split(" at ")[1].split(" with ")[0]
                revenue = txt.split(" with Rs.")[1].split(" Cr")[0]
                revenue_val = float(revenue.replace(",", "").strip())
                fallback_data.append({
                    "Startup Name": name.strip(),
                    "Industry": industry.strip(),
                    "Headquarters": hq.strip(),
                    "Total Revenue": revenue_val
                })
            except:
                continue

        if not fallback_data:
            st.warning("Could not parse retrieved data.")
            return

        fallback_df = pd.DataFrame(fallback_data).sort_values(by="Total Revenue", ascending=False).head(5).reset_index(drop=True)

        st.markdown("## 🤖 AI Insight (via Semantic Search)")
        for i, row in fallback_df.iterrows():
            display_card(row["Startup Name"], row["Industry"], format_revenue(row["Total Revenue"]), row["Headquarters"], i)
