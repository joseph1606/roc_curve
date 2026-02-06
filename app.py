import streamlit as st
import pandas as pd
from functions import calculate_roc_metrics, create_boxplot

st.title("ROC Curve & Boxplot Analysis")

uploaded_file = st.file_uploader("Excel file", type='xlsx')

if uploaded_file is not None:
    sheet = st.text_input("Sheet name", key="Sheet Name")
    
    if sheet:
        try:
            df = pd.read_excel(uploaded_file, sheet_name=sheet)
            st.dataframe(df)
            
            csv = df.to_csv(index=False)
            st.download_button("Download Data", csv, "data.csv", "text/csv")
            
            cases_col = st.text_input("Cases column", key="case")
            controls_col = st.text_input("Controls column", key="control")
            experiment_name = st.text_input("Experiment name", key="experiment")
            
            analysis_type = st.radio("Select Analysis:", ["ROC Curve", "Boxplot"])
            
            if cases_col and controls_col:
                if cases_col not in df.columns:
                    st.error(f"Column {cases_col} not found")
                elif controls_col not in df.columns:
                    st.error(f"Column {controls_col} not found")
                elif cases_col in df.columns and controls_col in df.columns:
                    cases = df[cases_col].dropna()
                    controls = df[controls_col].dropna()
                    
                    if len(cases) == 0:
                        st.error(f"No data in {cases_col}")
                    elif len(controls) == 0:
                        st.error(f"No data in {controls_col}")
                    elif len(cases) < 2 or len(controls) < 2:
                        st.error(f"Need at least 2 values in each column")
                    elif len(cases) >= 2 and len(controls) >= 2:
                        if analysis_type == "ROC Curve":
                            calculate_roc_metrics(cases, controls, experiment_name)
                        elif analysis_type == "Boxplot":
                            create_boxplot(cases, controls, cases_col, controls_col, experiment_name)
            elif cases_col or controls_col:
                st.warning("Enter both column names")
                
        except Exception as e:
            st.error(f"Error: {e}")