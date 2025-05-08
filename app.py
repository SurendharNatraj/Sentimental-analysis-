import streamlit as st
import pandas as pd

st.title("📄 CSV Viewer in Streamlit")

# File uploader (optional if you want to select files manually)
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("CSV file loaded successfully!")

    # Show data preview
    st.subheader("🔍 Data Preview")
    st.dataframe(df)

    # Optional: Show summary
    st.subheader("📊 Summary Statistics")
    st.write(df.describe())

    # Optional: Select column to visualize
    selected_column = st.selectbox("Select a column to visualize", df.columns)
    st.bar_chart(df[selected_column].value_counts())
