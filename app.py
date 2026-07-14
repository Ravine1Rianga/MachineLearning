import gradio as gr
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import gdown
import os

FILE_ID = "1as2FCB2iflnNk2lJS44b2bjHcE-L-CvR"

# --- Data Loading and Cleaning ---
def load_data():
    if not os.path.exists("Data_Visualization.txt"):
        gdown.download(f"https://drive.google.com/uc?id={FILE_ID}", "Data_Visualization.txt", quiet=False)
    df = pd.read_csv("Data_Visualization.txt", sep=None, engine='python', encoding='latin1')
    df = df.dropna(axis=1, how='all')
    df = df.loc[:, (df != 0).any(axis=0)]
    return df

try:
    df_clean = load_data()
    numeric_df = df_clean.select_dtypes(include=['float64', 'int64'])
    num_cols = numeric_df.columns.tolist()
except Exception as e:
    df_clean = pd.DataFrame()
    numeric_df = pd.DataFrame()
    num_cols = []

# --- Plotting Functions ---
def plot_heatmap():
    fig, ax = plt.subplots(figsize=(14, 10))
    if not numeric_df.empty:
        top_features = numeric_df.iloc[:, :20]
        sns.heatmap(top_features.corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, ax=ax)
    plt.tight_layout()
    return fig

def plot_boxplot(selected_elements):
    fig, ax = plt.subplots(figsize=(15, 6))
    if selected_elements and not numeric_df.empty:
        sns.boxplot(data=numeric_df[selected_elements], ax=ax)
        ax.set_ylabel("Percentage (%)")
    plt.tight_layout()
    return fig

def plot_distributions(chosen):
    if not chosen:
        fig, ax = plt.subplots()
        return fig
    
    rows = (len(chosen) + 1) // 2
    fig, axes = plt.subplots(rows, 2, figsize=(14, 5 * rows))
    axes = axes.flatten() if len(chosen) > 1 else [axes]
    
    for i, element in enumerate(chosen):
        sns.histplot(data=numeric_df, x=element, bins=30, kde=True, ax=axes[i], color='teal')
        axes[i].set_title(f'Distribution of {element}')
        
    # Remove unused axes
    for j in range(len(chosen), len(axes)):
        fig.delaxes(axes[j])
        
    plt.tight_layout()
    return fig

def plot_scatter(x_axis, y_axis):
    fig, ax = plt.subplots(figsize=(10, 6))
    if x_axis and y_axis and not numeric_df.empty:
        sns.scatterplot(data=numeric_df, x=x_axis, y=y_axis, alpha=0.5, color='darkorange', ax=ax)
        ax.set_title(f'{y_axis} vs. {x_axis}')
    plt.tight_layout()
    return fig

# --- Gradio UI Layout ---
with gr.Blocks(title="Metal Alloys Dashboard", theme=gr.themes.Soft()) as app:
    gr.Markdown("# Metal Alloys Visualization Dashboard")

    if df_clean.empty:
        gr.Markdown("### ❌ Could not load dataset.")
    else:
        with gr.Row():
            # SIDEBAR EQUIVALENT
            with gr.Column(scale=1):
                gr.Markdown("### Dataset Info")
                gr.Markdown(f"**Rows:** {df_clean.shape[0]}\n\n**Columns:** {df_clean.shape[1]}")
                gr.Dataframe(df_clean.head(), interactive=False)

            # MAIN TABS EQUIVALENT
            with gr.Column(scale=3):
                with gr.Tabs():
                    
                    # Tab 1: Heatmap
                    with gr.TabItem("Correlation Heatmap"):
                        gr.Markdown("### Correlation Heatmap (first 20 numeric columns)")
                        heatmap_output = gr.Plot()
                        app.load(plot_heatmap, inputs=[], outputs=heatmap_output)

                    # Tab 2: Boxplot
                    with gr.TabItem("Element Boxplot"):
                        gr.Markdown("### Spread and Outliers of Key Metal Concentrations")
                        default_box_cols = num_cols[6:12] if len(num_cols) > 6 else num_cols
                        box_dropdown = gr.Dropdown(choices=num_cols, value=default_box_cols, multiselect=True, label="Select elements:")
                        boxplot_output = gr.Plot()
                        
                        box_dropdown.change(plot_boxplot, inputs=box_dropdown, outputs=boxplot_output)
                        app.load(plot_boxplot, inputs=box_dropdown, outputs=boxplot_output)

                    # Tab 3: Distributions
                    with gr.TabItem("Distributions"):
                        gr.Markdown("### Distribution of Key Elements")
                        cols_present = [c for c in ['Al', 'Si', 'Cu', 'Ni'] if c in num_cols]
                        dist_default = cols_present or num_cols[:4]
                        dist_dropdown = gr.Dropdown(choices=num_cols, value=dist_default, multiselect=True, label="Elements to plot:")
                        dist_output = gr.Plot()
                        
                        dist_dropdown.change(plot_distributions, inputs=dist_dropdown, outputs=dist_output)
                        app.load(plot_distributions, inputs=dist_dropdown, outputs=dist_output)

                    # Tab 4: Scatter Explorer
                    with gr.TabItem("Scatter Explorer"):
                        gr.Markdown("### Cross-Property Relationship Explorer")
                        with gr.Row():
                            x_drop = gr.Dropdown(choices=num_cols, value=num_cols[0] if num_cols else None, label="X-Axis Feature")
                            y_drop = gr.Dropdown(choices=num_cols, value=num_cols[min(1, len(num_cols)-1)] if num_cols else None, label="Y-Axis Feature")
                        scatter_output = gr.Plot()
                        
                        x_drop.change(plot_scatter, inputs=[x_drop, y_drop], outputs=scatter_output)
                        y_drop.change(plot_scatter, inputs=[x_drop, y_drop], outputs=scatter_output)
                        app.load(plot_scatter, inputs=[x_drop, y_drop], outputs=scatter_output)

if __name__ == "__main__":
    app.launch()
