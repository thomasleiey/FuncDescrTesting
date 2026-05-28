import streamlit as st
import Home
import Settings

st.set_page_config(page_title="FunctionMap AI", layout="wide")

# Sidebar navigation
st.sidebar.title("🔧 Navigation")
page = st.sidebar.radio("Go to", ["🏠 Home", "⚙️ Settings"])

# Route pages
if page == "🏠 Home":
    Home.show()
elif page == "⚙️ Settings":
    Settings.show()
