import streamlit as st
import os
import json
import pandas as pd
import sqlite3

DB_FILE = "uploaded_data.db"
TABLE_NAME = "uploaded_keywords"
SETTINGS_FILE = ".settings.json"
OPENAI_KEY_FILE = ".openai_key.txt"

# --- File Upload Handler ---
def process_file_upload():
    st.markdown("---")


    uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx"])

    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)

            if df.shape[1] < 3:
                st.error("❌ File must contain at least 3 columns.")
                return

            df = df.iloc[:, :3]
            df.columns = ["LegacyFunctionPath", "MFH_Mapping", "NewFunctionDescription"]
            df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
            #df.dropna(subset=["LegacyFunctionPath", "MFH_Mapping", "NewFunctionDescription"], inplace=True)

            with sqlite3.connect(DB_FILE) as conn:
                df.to_sql(TABLE_NAME, conn, if_exists='replace', index=False)

            st.success("✅ Data uploaded and saved successfully!")

        except Exception as e:
            st.error(f"❌ Error processing file: {e}")

# --- Uploaded Data Viewer ---
def display_uploaded_data():
    df = get_uploaded_data_df()
    if df.empty:
        st.info("ℹ️ No data found. Please upload a file first.")
    else:
        st.dataframe(df.style.set_properties(**{
            'text-align': 'left',
            'white-space': 'pre-wrap'
        }))

# --- Getter for Uploaded Data ---
def get_uploaded_data_df() -> pd.DataFrame:
    try:
        with sqlite3.connect(DB_FILE) as conn:
            return pd.read_sql(f"SELECT * FROM {TABLE_NAME}", conn)
    except Exception as e:
        st.warning(f"⚠️ Could not fetch table data: {e}")
        return pd.DataFrame(columns=["LegacyFunctionPath", "MFH_Mapping", "NewFunctionDescription"])

# --- Key Management ---
def save_openai_key(key: str):
    with open(OPENAI_KEY_FILE, "w") as f:
        f.write(key.strip())

def load_openai_key():
    if os.path.exists(OPENAI_KEY_FILE):
        with open(OPENAI_KEY_FILE, "r") as f:
            return f.read().strip()
    return ""

def get_openai_key():
    return load_openai_key()

def get_special_requirements() -> str:
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, "r") as f:
                settings = json.load(f)
                return settings.get("special_requirements", "").strip()
        except Exception as e:
            st.warning(f"⚠️ Failed to load special requirements: {e}")
    return ""

def get_selected_llm() -> str:
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, "r") as f:
                settings = json.load(f)
                return settings.get("preferred_llm", "openai")
        except:
            pass
    return "openai"

def get_google_api_key() -> str:
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, "r") as f:
                settings = json.load(f)
                return settings.get("google_api_key", "").strip()
        except:
            pass
    return ""


# --- Settings UI ---
def show():
    st.set_page_config(page_title="Settings | Hierarchy Insight Pro", layout="centered")
    st.title("⚙️ Settings")
    st.markdown("Manage your API Keys, LLM Preferences, and Upload Mapping Data:")

    # Load existing settings
    existing_openai_key = load_openai_key()
    google_key = ""
    selected_llm = "openai"
    special_reqs = ""

    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, "r") as f:
                settings = json.load(f)
                google_key = settings.get("google_api_key", "")
                selected_llm = settings.get("preferred_llm", "openai")
                special_reqs = settings.get("special_requirements", "")
        except Exception as e:
            st.warning(f"⚠️ Failed to load settings: {e}")
    st.markdown("""
<style>
/* Hide the whole toggle container on most Streamlit builds */
[data-testid="stPasswordInput"] > div > div:nth-child(2) { 
    display: none !important;
}

/* Fallbacks for other builds */
[data-testid="stPasswordInput"] svg,
[data-testid="stPasswordInput"] button,
[data-testid="stPasswordInput"] div[role="button"],
[data-testid="stTextInput"] [aria-label*="password"] {
    display: none !important;
}

/* Remove the extra right padding once the button is gone */
[data-testid="stPasswordInput"] input {
    padding-right: 0 !important;
}
</style>
""", unsafe_allow_html=True)
    # --- Form: API Keys & LLM Preference ---
    with st.form("llm_settings_form"):
        st.markdown("### 🔐 API Keys & Model Selection")

        openai_key = st.text_input(
            "OpenAI API Key",
            type="password",
            value=existing_openai_key,
            placeholder="Enter your OpenAI API key..."
        )

        google_api_key = st.text_input(
            "Google AI API Key (Gemini)",
            type="password",
            value=google_key,
            placeholder="Enter your Google API key..."
        )

        preferred_llm = st.selectbox("Preferred LLM Provider", options=["openai", "google"], index=["openai", "google"].index(selected_llm))

        submitted = st.form_submit_button("💾 Save LLM Settings")
        if submitted:
            try:
                # Save OpenAI key to file
                if openai_key:
                    save_openai_key(openai_key)

                # Save settings
                updated_settings = {
                    "preferred_llm": preferred_llm,
                    "google_api_key": google_api_key.strip(),
                    "special_requirements": special_reqs.strip()  # Keep unchanged for now
                }

                with open(SETTINGS_FILE, "w") as f:
                    json.dump(updated_settings, f, indent=2)

                st.success("✅ LLM settings saved successfully!")
            except Exception as e:
                st.error(f"❌ Failed to save settings: {e}")

    # --- Upload File ---
    process_file_upload()

    # --- Display Uploaded Data ---
    display_uploaded_data()

    # --- Form: Special LLM Prompt Requirements ---
    st.markdown("### ✍️ Special Requirements for LLM Prompt Behavior")
    with st.form("special_requirements_form"):
        special_input = st.text_area(
            "Enter specific instructions for LLM prompts:",
            value=special_reqs,
            height=150,
            placeholder="E.g. Always follow MFH mapping strictly, avoid generic placeholders like XX, etc."
        )

        req_submit = st.form_submit_button("🔄 Update Requirements")
        if req_submit:
            try:
                # Reload current settings
                current = {}
                if os.path.exists(SETTINGS_FILE):
                    with open(SETTINGS_FILE, "r") as f:
                        current = json.load(f)

                # Update and write back
                current["special_requirements"] = special_input.strip()
                with open(SETTINGS_FILE, "w") as f:
                    json.dump(current, f, indent=2)

                st.success("✅ Requirements updated successfully!")
            except Exception as e:
                st.error(f"❌ Failed to save requirements: {e}")
