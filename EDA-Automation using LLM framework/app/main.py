import gradio as gr
import pandas as pd

from app.eda import generate_eda_context
from app.llm_insights import generate_llm_insights
from app.stats_human_baseline import (
    human_statistical_summary,
    human_summary_text
)


def eda_analysis(file_path):
    df = pd.read_csv(file_path)

    # Human statistical baseline
    human_results = human_statistical_summary(df)
    human_text = human_summary_text(human_results)

    # Structured EDA context
    eda_context = generate_eda_context(df)

    # LLM-based insights
    llm_text = generate_llm_insights(eda_context)

    report = f"""
===========================
HUMAN STATISTICAL ANALYSIS
===========================
{human_text}

===========================
LLM-GENERATED INSIGHTS
===========================
{llm_text}
"""

    # Generate plots
    plots = []
    numeric_cols = df.select_dtypes(include=["number"]).columns

    for col in numeric_cols:
        fig = df[col].hist().get_figure()
        plot_path = f"{col}_distribution.png"
        fig.savefig(plot_path)
        plots.append(plot_path)
        fig.clf()

    return report, plots


demo = gr.Interface(
    fn=eda_analysis,
    inputs=gr.File(type="filepath"),
    outputs=[
        gr.Textbox(
            label="EDA Report (Human + LLM)",
            lines=30,
            max_lines=50
        ),
        gr.Gallery(
            label="Data Visualizations",
            columns=2,
            height="auto"
        )
    ],
    title="Evaluating LLMs for Automated Exploratory Data Analysis",
    description=(
        "Upload a CSV dataset to compare statistically rigorous human EDA "
        "with LLM-generated analytical insights."
    )
)

if __name__ == "__main__":
    demo.launch()

