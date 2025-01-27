import streamlit as st
from tools import load_data

st.set_page_config(page_title="Breast Cancer", layout="wide")
st.title("Breast Cancer Wisconsin")

# Carregar dados
# Load data into session state
if "conn" not in st.session_state or "df" not in st.session_state:
    df, conn, variable_info = load_data()
    st.session_state["df"] = df
    st.session_state["conn"] = conn
    st.session_state["variable_info"] = variable_info

Overview = st.Page("overview.py", title="Overview", icon=":material/flag:", default=True)

eda = st.Page("exploratory.py", title="Exploratory Data Analysis", icon=":material/troubleshoot:")
feature_eng = st.Page("feat_eng.py", title="Feature engineering", icon=":material/troubleshoot:")

modeling = st.Page("models.py", title="Modeling", icon="🧠")

pg = st.navigation(
    {
        "Overview": [Overview],
        "EDA": [eda, feature_eng],
        "Models": [modeling],
    }
)
pg.run()
