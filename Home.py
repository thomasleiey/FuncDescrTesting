import streamlit as st
import pandas as pd
import re
from Settings import get_openai_key
from newFuntionDescriptionPredictor import newFuntionDescriptionPredictor 

def show():
    st.title("🧠 FunctionMap AI")
    st.markdown("##### Predict & Refine Functional Descriptions with MFH Mapping Intelligence")
    st.markdown("---")

    # --- Upload File ---
    uploaded_file = st.file_uploader(
    "📤 Upload Excel or CSV file with columns: `FuncDescr`, `legacyfunctionidpath`, and `MFHMapping`",
    type=["xlsx", "csv"]
)


    # Process uploaded file once
    if uploaded_file and "uploaded_filename" not in st.session_state:
        # Clear previous session state
        for key in [
        "df", "phrase_groups", "processed_df", "edited_df",
        "processed_keywords", "process_triggered", "processing_required", "final_editor"
    ]:
          st.session_state.pop(key, None)


        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(".xlsx"):
            df = pd.read_excel(uploaded_file)
        else:
            st.error("❌ Unsupported file type. Please upload a .csv or .xlsx file.")
            return None
        
        phrase_groups = df["MFHMapping"].dropna().unique()
        st.session_state["df"] = df
        st.session_state["phrase_groups"] = phrase_groups
        st.session_state["uploaded_filename"] = uploaded_file.name

    if "df" in st.session_state and "phrase_groups" in st.session_state:
        df = st.session_state["df"]
        phrase_groups = st.session_state["phrase_groups"]

        

        st.markdown("#### 🎯 Select SME Keyword Groups")
        selected_display_labels = st.multiselect(
            "Step 1: Choose group(s):",
            options=phrase_groups
        )

        
                # Set this once on first run
        if "process_triggered" not in st.session_state:
            st.session_state.process_triggered = False
        if "processing_required" not in st.session_state:
            st.session_state.processing_required = False

        # When button is clicked
        if st.button("🔍 Process"):
            if not selected_display_labels:
                st.warning("⚠️ Please select at least one group before processing.")
            else:
                st.session_state.processing_required = True
                st.session_state.process_triggered = True
                st.session_state.processed_keywords = selected_display_labels
                
                # ✅ Clear previous results to allow reprocessing
                for key in ["edited_df", "new_df"]:
                    st.session_state.pop(key, None)


        if st.session_state.get("processing_required", False):
            if "new_df" not in st.session_state:
                # Only run prediction once
                selected_keywords = [kw.strip().lower() for kw in st.session_state["processed_keywords"]]
                filtered_df = df[df["MFHMapping"].isin(st.session_state["processed_keywords"])]

                predictor = newFuntionDescriptionPredictor(openai_api_key=get_openai_key(), batch_size=25)
                progress_bar = st.progress(0)
                status_text = st.empty()

                def update_progress(current, total):
                    pct = min(current / total, 1.0)
                    progress_bar.progress(pct)
                    status_text.text(f"Processed {current} / {total} rows...")

                # ✅ Run only once
                new_df = predictor.predict_dataframe(filtered_df, streamlit_callback=update_progress)
                st.session_state["new_df"] = new_df
                st.success("✅ New Function Description Prediction Complete!")
            else:
                new_df = st.session_state["new_df"]
                st.success("✅ Showing Previously Predicted Data")

            # ✅ Show editable table
            edited_df = st.data_editor(new_df, use_container_width=True, key="edited_df_key")
            st.session_state["edited_df"] = edited_df

            # ✅ Optional download
            csv = edited_df.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download Edited Data", data=csv, file_name="edited_output.csv", mime="text/csv")


            
           

    else:
        st.info("📤 Please upload an Excel file to begin.")
