
import pandas as pd
import numpy as np
import scipy.stats as stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


def human_statistical_summary(df: pd.DataFrame) -> dict:
    """
    Returns structured results used as ground truth.
    """

    results = {}

#1. Descriptive Statistics
    mean_units = df["units_sold"].mean()
    std_units = df["units_sold"].std()

    results["mean_units_sold"] = mean_units
    results["std_units_sold"] = std_units

#2. Confidence Interval
    ci_low, ci_high = stats.t.interval(
        0.95,
        len(df["units_sold"]) - 1,
        loc=mean_units,
        scale=stats.sem(df["units_sold"])
    )
    results["ci_95"] = (ci_low, ci_high)

#3. Hypothesis Testing
#H0: mean == 20
    t_stat, p_value = stats.ttest_1samp(df["units_sold"], 20)
    results["t_stat"] = t_stat
    results["p_value"] = p_value
    results["reject_null"] = p_value < 0.05

#4. Effect Size (Cohen's d)
    cohens_d = (mean_units - 20) / std_units
    results["cohens_d"] = cohens_d

#5. Interpretable ML: Linear Regression
#Predict units_sold using revenue (simple, explainable model)
    if "revenue" in df.columns:
        X = df[["revenue"]].values
        y = df["units_sold"].values

        model = LinearRegression()
        model.fit(X, y)
        y_pred = model.predict(X)

        results["regression_coef"] = model.coef_[0]
        results["regression_intercept"] = model.intercept_
        results["r2_score"] = r2_score(y, y_pred)

#6. Category-level Variance
    if "category" in df.columns:
        cat_var = df.groupby("category")["units_sold"].var()
        results["highest_variance_category"] = cat_var.idxmax()

    return results


def human_summary_text(results: dict) -> str:
    """
    Converts structured statistical results into a human-readable summary.
    """

    summary = []

    summary.append(
        f"Average units sold is {results['mean_units_sold']:.2f} "
        f"(std = {results['std_units_sold']:.2f})."
    )

    summary.append(
        f"95% confidence interval for mean sales is "
        f"({results['ci_95'][0]:.2f}, {results['ci_95'][1]:.2f})."
    )

    if results["reject_null"]:
        summary.append(
            "Hypothesis testing indicates the mean sales significantly differ from 20."
        )
    else:
        summary.append(
            "Hypothesis testing fails to reject that mean sales equal 20."
        )

    summary.append(
        f"Effect size (Cohen's d) is {results['cohens_d']:.2f}, "
        "indicating practical impact."
    )

    if "r2_score" in results:
        summary.append(
            f"Linear regression shows revenue explains "
            f"{results['r2_score']:.2%} of variance in units sold."
        )

    summary.append(
        f"The {results['highest_variance_category']} category shows highest variability."
    )

    return " ".join(summary)
