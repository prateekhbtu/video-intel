import streamlit as st
import psycopg2
import pandas as pd
import os
import json

st.set_page_config(page_title="Video Intel Hub", layout="wide")
st.title("📹 Video Intelligence: Live Fleet Dashboard")

DB_URL = os.getenv("DATABASE_URL")

# Cache data for 2 seconds to avoid spamming the database
@st.cache_data(ttl=2)
def fetch_data():
    try:
        conn = psycopg2.connect(DB_URL)
        query = "SELECT camera_id, site_id, created_at, payload FROM events ORDER BY created_at DESC LIMIT 200;"
        df = pd.read_sql(query, conn)
        conn.close()
        return df
    except Exception as e:
        st.error(f"DB Connection Error: {e}")
        return pd.DataFrame()

df = fetch_data()

if not df.empty:
    # Extract the tracked object count from the JSON payload
    df['tracked_objects'] = df['payload'].apply(lambda x: x.get('count', 0) if isinstance(x, dict) else 0)
    df.set_index('created_at', inplace=True)
    
    # Top level metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Events Processed (last 200)", len(df))
    col2.metric("Active Cameras", df['camera_id'].nunique())
    col3.metric("Peak Objects Tracked", df['tracked_objects'].max())

    st.markdown("---")
    
    # Live Chart
    st.subheader("Live Object Detections (Per Camera)")
    chart_data = df.pivot_table(index='created_at', columns='camera_id', values='tracked_objects', aggfunc='max').fillna(0)
    st.line_chart(chart_data)

    # Raw Data Table
    st.subheader("Recent Inference Events")
    st.dataframe(df[['site_id', 'camera_id', 'tracked_objects']].head(10))
    
    # Auto-refresh trigger
    st.button("🔄 Refresh Data")
else:
    st.warning("Waiting for data from the Cloud API...")
