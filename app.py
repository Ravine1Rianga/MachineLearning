import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import gdown
import os

FILE_ID = "1SglbaHOdnWHxal2aEAfdf02NVzH2Pw79"

st.set_page_config(layout="wide", page_title=" Alloys of Metals Dashboard")
st.title("Alloys of Metal Dashboard Visualization")

@st.cache_data
def load_data():
    if not os.path.exists("Data_Visualization.txt"):
        gdown.download(f"https://drive.google.com/uc?id={FILE_ID}", "Data_Visualization.txt", quiet=False)
    df = pd.read_csv("Data_Visualization.txt", sep=None, engine='python', encoding='latin1')
    df = df.dropna(axis=1, how='all')
    df = df.loc[:, (df != 0).any(axis=0)]
    return df

df_clean = load_data()

if df_clean.empty:
    st.error("Could not load dataset.")
else:
    numeric_df = df_clean.select_dtypes(include=['float64', 'int64'])

    st.sidebar.header("Dataset Info")
    st.sidebar.write(f"Rows: {df_clean.shape[0]}")
    st.sidebar.write(f"Columns: {df_clean.shape[1]}")
    st.sidebar.dataframe(df_clean.head())

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Correlation Heatmap", "Element Boxplot", "Distributions", "Scatter Explorer"]
    )

    with tab1:
        st.subheader("Correlation Heatmap (first 20 numeric columns)")
        top_features = numeric_df.iloc[:, :20]
        if not top_features.empty:
            fig1, ax1 = plt.subplots(figsize=(14, 10))
            sns.heatmap(top_features.corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, ax=ax1)
            st.pyplot(fig1)
            plt.close(fig1)

    with tab2:
        st.subheader("Spread and Outliers of Key Metal Concentrations")
        default_cols = list(numeric_df.columns[6:12]) if numeric_df.shape[1] > 6 else list(numeric_df.columns)
        selected_elements = st.multiselect("Select elements:", numeric_df.columns.tolist(), default=default_cols)
        if selected_elements:
            fig2, ax2 = plt.subplots(figsize=(15, 6))
            sns.boxplot(data=numeric_df[selected_elements], ax=ax2)
            ax2.set_ylabel("Percentage (%)")
            st.pyplot(fig2)
            plt.close(fig2)

    with tab3:
        st.subheader("Distribution of Key Elements")
        cols_present = [c for c in ['Al', 'Si', 'Cu', 'Ni'] if c in numeric_df.columns]
        chosen = st.multiselect("Elements to plot:", numeric_df.columns.tolist(),
                                 default=cols_present or numeric_df.columns[:4].tolist())
        if chosen:
            rows = (len(chosen) + 1) // 2
            fig3, axes = plt.subplots(rows, 2, figsize=(14, 5 * rows))
            axes = axes.flatten() if len(chosen) > 1 else [axes]
            for i, element in enumerate(chosen):
                sns.histplot(data=numeric_df, x=element, bins=30, kde=True, ax=axes[i], color='teal')
                axes[i].set_title(f'Distribution of {element}')
            for j in range(len(chosen), len(axes)):
                fig3.delaxes(axes[j])
            plt.tight_layout()
            st.pyplot(fig3)
            plt.close(fig3)

    with tab4:
        st.subheader("Cross-Property Relationship Explorer")
        col1, col2 = st.columns(2)
        with col1:
            x_axis = st.selectbox("X-Axis Feature:", numeric_df.columns.tolist(), index=0)
        with col2:
            y_axis = st.selectbox("Y-Axis Feature:", numeric_df.columns.tolist(), index=min(1, len(numeric_df.columns)-1))
        fig4, ax4 = plt.subplots(figsize=(10, 6))
        sns.scatterplot(data=numeric_df, x=x_axis, y=y_axis, alpha=0.5, color='darkorange', ax=ax4)
        ax4.set_title(f'{y_axis} vs. {x_axis}')
        st.pyplot(fig4)
        plt.close(fig4)
