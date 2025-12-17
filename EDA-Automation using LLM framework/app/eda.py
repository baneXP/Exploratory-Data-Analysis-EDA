
import pandas as pd

def generate_eda_context(df: pd.DataFrame) -> dict:
    
    #Generates structured EDA facts to be consumed by an LLM.

    context = {}

# Numeric column statistics
    numeric_cols = df.select_dtypes(include=["number"]).columns
    numeric_summary = {}

    for col in numeric_cols:
        numeric_summary[col] = {
            "mean": float(df[col].mean()),
            "median": float(df[col].median()),
            "std": float(df[col].std()),
            "min": float(df[col].min()),
            "max": float(df[col].max())
        }

    context["numeric_summary"] = numeric_summary

# Category-wise statistics
    if "category" in df.columns and "units_sold" in df.columns:
        context["category_stats"] = (
            df.groupby("category")["units_sold"]
              .agg(["mean", "std", "var"])
              .to_dict()
        )

# Correlation (important for EDA)
    if len(numeric_cols) > 1:
        context["correlation_matrix"] = (
            df[numeric_cols]
            .corr()
            .round(3)
            .to_dict()
        )
# Dataset shape
    context["dataset_shape"] = {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1])
    }

    return context
